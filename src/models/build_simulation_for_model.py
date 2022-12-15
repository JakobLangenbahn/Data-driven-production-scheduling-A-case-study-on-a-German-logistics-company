""" Define functions for creating models and the required simulations """
from datetime import datetime
from math import sqrt
from random import sample

import simpy
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import MDPInfo, Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.spaces import Discrete
from tensorforce import Environment
from mushroom_rl.core import Environment as EnvironmentMushroom
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.models.train_models import train_som
from src.simulation import Plant, select_machine_winq
from src.utils import transform_to_real_time


class SelectEnvironment(Environment):
    """
    Define a simulation environment for training and evaluating reinforcement algorithms which select dispatching rules
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machines for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param simulation_start: Simulation start in clock time of real world
    :param priority_rules: Set of priority rule for dispatching decision
    :param allocation_rule: Allocation rule for dispatching decision
    :param random_state: Random state for reproducibility
    """

    def __init__(self, product_types_df, machines_df, orders_df, simulation_start, priority_rules, allocation_rule,
                 random_state):
        """
        Constructor method
        """
        super().__init__()

        self.product_types_df = product_types_df
        self.machines_df = machines_df
        self.orders_df = orders_df
        self.simulation_start = simulation_start
        self.average_count_new_orders = None
        self.number_orders_start = None
        self.random_state = None
        self.worker = None
        self.due_date_range = None
        self.priority_rules = priority_rules
        self.priority_rule = sample(self.priority_rules, 1)[0]
        self.allocation_rule = allocation_rule
        self.runs = 0
        self.env = None
        self.plant = None
        self.random_state = random_state

    def states(self):
        """
        Define dimension of states
        """
        return dict(type='float', shape=(52,))

    def actions(self):
        """
        Define dimension of actions
        """
        return dict(type='int', num_values=8)

    def max_episode_timesteps(self):
        """
        Define maximum of time steps
        """
        return super().max_episode_timesteps()

    def close(self):
        """
        Method to close environment
        """
        super().close()

    def reset(self):
        """
        Method to reset the simulation environment
        :returns: State after environment is reset
        """
        self.runs += 1
        self.env = simpy.Environment()
        self.priority_rule = sample(self.priority_rules, 1)[0]
        self.plant = Plant(self.env, self.product_types_df, self.machines_df,
                           self.orders_df, self.average_count_new_orders, self.number_orders_start,
                           self.simulation_start,
                           self.priority_rule, self.allocation_rule, self.random_state, self.worker,
                           self.due_date_range)
        self.env.run(24 * 3600)
        self.plant.calculate_metrics()
        self.plant.calculate_state()
        state = list(self.plant.state.values())
        return state

    def execute(self, actions):
        """
        Execute action in simulation environment
        :param actions: Actions chosen by learning algorithm
        :returns: Next state of environment, flag if it was the last step of simulation and the resulting reward
        """
        self.plant.priority_rule = self.priority_rules[actions]
        sim_time_now = self.env.now
        self.env.run(sim_time_now + 24 * 3600)
        self.plant.calculate_metrics()
        self.plant.calculate_state()
        next_state = list(self.plant.state.values())
        reward = self.plant.revenue_today - self.plant.penalty_today
        if sim_time_now < 30 * 24 * 3600:
            terminal = False
        else:
            terminal = True
        return next_state, terminal, reward


class AssignEnvironment(Environment):
    """
    Define a simulation environment for training and evaluating reinforcement algorithms which directly assigning jobs
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machines for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param simulation_start: Simulation start in clock time of real world
    :param priority_rule: Priority rule for dispatching decision
    :param allocation_rule: Allocation rule for dispatching decision
    :param random_state: Random state for reproducibility
    """

    def __init__(self, product_types_df, machines_df, orders_df, simulation_start, priority_rule, allocation_rule,
                 random_state):
        """
        Constructor method
        """
        super().__init__()

        self.product_types_df = product_types_df
        self.machines_df = machines_df
        self.orders_df = orders_df
        self.simulation_start = simulation_start
        self.average_count_new_orders = None
        self.number_orders_start = None
        self.random_state = None
        self.worker = None
        self.due_date_range = None
        self.priority_rule = priority_rule
        self.allocation_rule = allocation_rule
        self.runs = 0
        self.env = None
        self.plant = None
        self.random_state = random_state

    def states(self):
        """
        Define dimension of states
        """
        return dict(type='float', shape=(173,))

    def actions(self):
        """
        Define dimension of actions
        """
        return dict(type='int', num_values=51)

    def max_episode_timesteps(self):
        """
        Define maximum of time steps
        """
        return super().max_episode_timesteps()

    def close(self):
        """
        Method to close environment
        """
        super().close()

    def reset(self):
        """
        Method to reset the simulation environment
        :returns: State after environment is reset
        """
        self.env = simpy.Environment()
        self.plant = Plant(self.env, self.product_types_df, self.machines_df,
                           self.orders_df, self.average_count_new_orders, self.number_orders_start,
                           self.simulation_start,
                           self.priority_rule, self.allocation_rule, self.random_state, self.worker,
                           self.due_date_range, dispatching=False)
        self.plant.calculate_metrics()
        self.plant.calculate_state()
        state = list(self.plant.state.values())
        # Select order set based on edd
        order_list = []
        for order in self.plant.uninitialized_orders:
            order_list.append({"Order": order,
                               "dt_due": order.due_date,
                               "pallets_planned": order.number_of_pallets})
        order_df = pd.DataFrame(order_list)
        if not order_df.empty:
            order_subset = order_df.sort_values(["dt_due", "pallets_planned"]).reset_index(drop=True)
            self.order_set = order_subset.Order[0:30].to_list()
        date_today = datetime.fromtimestamp(
            transform_to_real_time(self.plant.env.now, self.plant.simulation_start) / 1000)
        time_of_day = self.env.now % (24 * 3600)
        for order in self.order_set:
            state.append(order.predicted_processing_time_total)
            state.append((order.due_date - date_today).seconds)
            state.append(order.product_type.product_type_id)
            state.append(order.total_revenue)
        state.append(time_of_day)
        return state

    def execute(self, actions):
        """
        Execute action in simulation environment
        :param actions: Actions chosen by learning algorithm
        :returns: Next state of environment, flag if it was the last step of simulation and the resulting reward
        """
        start_action = self.env.now
        # if action 50 then no orders is selected and waited
        if actions < 30 and self.order_set:
            order_to_activate = self.order_set[actions]
            order_to_activate.initialize()
            self.plant.uninitialized_orders.remove(order_to_activate)
        self.env.run(until=self.plant.step_event)
        end_action = self.env.now

        penalty = self.plant.penalty_step
        revenue = self.plant.revenue_step
        reward = revenue - penalty

        self.plant.calculate_metrics()

        self.plant.calculate_state()
        state = list(self.plant.state.values())
        date_today = datetime.fromtimestamp(
            transform_to_real_time(self.env.now, self.simulation_start) / 1000)
        order_list = []
        for order in self.plant.uninitialized_orders:
            order_list.append({"Order": order,
                               "dt_due": order.due_date,
                               "pallets_planned": order.number_of_pallets})
        order_df = pd.DataFrame(order_list)
        if order_df.shape[0] < 30:
            order_df = order_df.sample(30, replace=True)
        order_subset = order_df.sort_values(["dt_due", "pallets_planned"]).reset_index(drop=True)
        if not order_df.empty:
            self.order_set = order_subset.Order[0:30].to_list()
            # Needed for markov property
            time_of_day = self.env.now % (24 * 3600)
            for order in self.order_set:
                state.append(order.predicted_processing_time_total)
                state.append((order.due_date - date_today).seconds)
                state.append(order.product_type.product_type_id)
                state.append(order.total_revenue)
        else:
            for i in range(50):
                state.append(0)
                state.append(0)
                state.append(0)
                state.append(0)
        state.append(time_of_day)

        next_state = state
        self.plant.penalty_step = 0
        self.plant.revenue_step = 0

        if self.env.now < 30 * 24 * 3600:
            terminal = False
        else:
            terminal = True
        return next_state, terminal, reward


def run_agent(agent, environment, due_date_range_list, number_orders_start_list, average_count_new_orders_list,
              worker_list, random_states, episodes=1, evaluate=False):
    """
    Train or evaluate reinforcement learning algorithm in simulation environment
    :param agent: Agent to train or evaluate
    :param environment: Simulation environment to train or evaluate the agent
    :param due_date_range_list: List of due date ranges for simulation
    :param number_orders_start_list: List of numbers orders start for simulation
    :param average_count_new_orders_list: List of average counts new orders for simulation
    :param worker_list: List of worker number for simulation
    :param random_states:List of random states for simulation
    :param episodes: Number of repetitions for training or evaluating the models
    :param evaluate: Flag if model should be trained or evaluated
    :returns: Dataframe including the rewards of the simulation period
    """
    reward_list = []
    iteration = 0
    for episode in tqdm(range(episodes)):
        for random_state in random_states:
            environment.random_state = random_state
            for due_date_range in due_date_range_list:
                for number_orders_start in number_orders_start_list:
                    for average_count_new_orders in average_count_new_orders_list:
                        for worker in worker_list:
                            # Change random state before each simulation to increase variety of data
                            environment.random_state += 1
                            iteration += 1
                            # Episode using act and observe
                            environment.due_date_range = due_date_range
                            environment.number_orders_start = number_orders_start
                            environment.average_count_new_orders = average_count_new_orders
                            environment.worker = worker
                            states = environment.reset()
                            terminal = False
                            num_updates = 0
                            day = 0
                            while not terminal:
                                day += 1
                                if evaluate:
                                    actions = agent.act(states=states, independent=True, deterministic=True)
                                    states, terminal, reward = environment.execute(actions=actions)
                                    num_updates += 1
                                else:
                                    actions = agent.act(states=states)
                                    states, terminal, reward = environment.execute(actions=actions)
                                    num_updates += agent.observe(terminal=terminal, reward=reward)
                                reward_list.append({"episode": episode,
                                                    "day": day,
                                                    "reward": reward,
                                                    "due_date_range": due_date_range,
                                                    "number_orders_start": number_orders_start,
                                                    "average_count_new_orders": average_count_new_orders,
                                                    "worker": worker,
                                                    "iteration": iteration
                                                    })
    return pd.DataFrame(reward_list)


class PlantEnv(EnvironmentMushroom):
    """
    Create environment for training and evaluating reinforcement algorithm based on mushroom rl package
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machines for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param simulation_start: Simulation start in clock time of real world
    :param priority_rules: Set of priority rule for dispatching decision
    :param allocation_rule: Allocation rule for dispatching decision
    :param random_state: Random state for reproducibility
    :param number_states: Number of cluster of self organizing map
    :param number_actions: Number of available actions
    :param som: Trained self organizing maps
    :param normalize_som: Function to normalize data before applying self organizing map
    """

    def __init__(self, product_types_df, machines_df, orders_df, simulation_start, priority_rules,
                 allocation_rule, random_state, number_states, number_actions, som, normalize_som):
        """
        Constructor method
        """
        # Save important environment information
        self.product_types_df = product_types_df
        self.machines_df = machines_df
        self.orders_df = orders_df
        self.simulation_start = simulation_start
        self.average_count_new_orders = None
        self.number_orders_start = None
        self.random_state = None
        self.worker = None
        self.due_date_range = None
        self.priority_rules = priority_rules
        self.priority_rule = sample(self.priority_rules, 1)[0]
        self.allocation_rule = allocation_rule
        self.runs = 0
        self.env = None
        self.plant = None
        self.random_state = random_state
        self.number_of_states = number_states
        self.som = som
        self.normalize_som = normalize_som

        # Create the action space.
        action_space = Discrete(number_actions)  # 4 actions: N, S, W, E

        # Create the observation space. It's a 2D box of dimension (size x size).
        # You can also specify low and high array, if every component has different limits
        observation_space = Discrete(number_states)

        # Create the MDPInfo structure, needed by the environment interface
        mdp_info = MDPInfo(observation_space=observation_space, action_space=action_space, gamma=0.99, horizon=100)
        super().__init__(mdp_info)

        self.size = mdp_info.size
        self.gamma = mdp_info.gamma
        self.horizon = mdp_info.horizon

        # Create a state class variable to store the current state
        self._state = None

    def reset(self, state=None):
        """
        Method to reset the simulation environment
        :param state: Optional start state
        :returns: State after environment is reset
        """
        self.runs += 1
        self.env = simpy.Environment()
        self.priority_rule = sample(self.priority_rules, 1)[0]
        self.plant = Plant(self.env, self.product_types_df, self.machines_df,
                           self.orders_df, self.average_count_new_orders, self.number_orders_start,
                           self.simulation_start,
                           self.priority_rule, self.allocation_rule, self.random_state, self.worker,
                           self.due_date_range)
        self.env.run(24 * 3600)
        self.plant.calculate_metrics()
        self.plant.calculate_state()
        df_state = self.plant.state
        for rule in self.priority_rules:
            df_state[rule.__name__] = int(self.plant.priority_rule.__name__ == rule.__name__)
        df = pd.DataFrame(df_state, index=[0])
        df_normalized = self.normalize_som(df)
        som_cluster = self.som.winner(df_normalized.values)
        state = int(som_cluster[0] * sqrt(self.number_of_states) + som_cluster[1])
        self._state = np.array([state])

        # Return the current state
        return self._state

    def step(self, action):
        """
        Execute action in simulation environment
        :param action: Action chosen by learning algorithm
        :returns: Next state of environment, the resulting reward and flag if it was the last step of simulation
        """
        self.plant.priority_rule = self.priority_rules[action[0]]
        sim_time_now = self.env.now
        self.env.run(sim_time_now + 24 * 3600)
        self.plant.calculate_metrics()
        self.plant.calculate_state()
        df_state = self.plant.state
        for rule in self.priority_rules:
            df_state[rule.__name__] = int(self.plant.priority_rule.__name__ == rule.__name__)
        df = pd.DataFrame(df_state, index=[0])
        df_normalized = self.normalize_som(df)
        som_cluster = self.som.winner(df_normalized.values)
        state = int(som_cluster[0] * sqrt(self.number_of_states) + som_cluster[1])
        reward = self.plant.revenue_today - self.plant.penalty_today
        if sim_time_now < 30 * 24 * 3600:
            absorbing = False
        else:
            absorbing = True
        self._state = np.array([state])

        # Return all the information + empty dictionary (used to pass additional information)
        return self._state, reward, absorbing, {}


# Register environment in mushroom package
PlantEnv.register()


def run_agent_mushroom(data, product_types_df, machines_df, orders_df, simulation_start, priority_rules,
                       due_date_range_list, number_orders_start_list, average_count_new_orders_list, worker_list,
                       random_states, params, episodes=1):
    """
    Train and evaluate reinforcement agent based on mushroom environment
    :param data: Dataset for training self organizing map
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machines for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param simulation_start: Simulation start in clock time of real world
    :param priority_rules: Set of priority rule for dispatching decision
    :param due_date_range_list: List of due date ranges for simulation
    :param number_orders_start_list: List of numbers orders start for simulation
    :param average_count_new_orders_list: List of average counts new orders for simulation
    :param worker_list: List of worker number for simulation
    :param random_states:List of random states for simulation
    :param params: Set of hyperparameter for training
    :param episodes: Number of repetitions for training or evaluating the models
    :returns: List of reward obtained in evaluation
    """
    # Setup environment
    som, normalize_som = train_som(params["number_of_states"], params["number_of_states"], data, params["sigma"],
                                   params["learning_rate_som"], 42, number_iterations=10000)
    env = EnvironmentMushroom.make('PlantEnv', product_types_df=product_types_df, machines_df=machines_df,
                                   orders_df=orders_df,
                                   simulation_start=simulation_start, priority_rules=priority_rules,
                                   allocation_rule=select_machine_winq, random_state=42,
                                   number_states=params["number_of_states"] * params["number_of_states"],
                                   number_actions=8,
                                   som=som, normalize_som=normalize_som)
    epsilon = Parameter(value=params["epsilon_param"])
    policy = EpsGreedy(epsilon=epsilon)

    learning_rate = Parameter(value=params["learning_rate"])
    agent = QLearning(env, policy, learning_rate)

    core = Core(agent, env)

    reward_list = []
    iteration = 0
    for _ in tqdm(range(episodes)):
        for random_state in random_states:
            env.random_state = random_state
            for due_date_range in due_date_range_list:
                for number_orders_start in number_orders_start_list:
                    for average_count_new_orders in average_count_new_orders_list:
                        for worker in worker_list:
                            # Change random state before each simulation to increase variety of data
                            env.random_state += 1
                            iteration += 1
                            # Episode using act and observe
                            env.due_date_range = due_date_range
                            env.number_orders_start = number_orders_start
                            env.average_count_new_orders = average_count_new_orders
                            env.worker = worker
                            core.learn(n_episodes=1, n_steps_per_fit=1, render=False, quiet=True)

    for _ in tqdm(range(episodes)):
        for random_state in random_states:
            env.random_state = random_state
            for due_date_range in due_date_range_list:
                for number_orders_start in number_orders_start_list:
                    for average_count_new_orders in average_count_new_orders_list:
                        for worker in worker_list:
                            # Change random state before each simulation to increase variety of data
                            env.random_state += 1
                            iteration += 1
                            # Episode using act and observe
                            env.due_date_range = due_date_range
                            env.number_orders_start = number_orders_start
                            env.average_count_new_orders = average_count_new_orders
                            env.worker = worker
                            reward_list.extend(core.evaluate(n_episodes=1, render=False, quiet=True))
    return reward_list
