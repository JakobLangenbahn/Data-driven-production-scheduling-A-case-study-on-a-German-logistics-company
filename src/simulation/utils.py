""" Util functions for simulation in modelling """
import simpy
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.simulation import select_machine_winq, assign_priority_edd
from src.simulation import Plant


def initialize_simulation(priority_rule_start, product_types_df, machines_df, orders_df, average_count_new_orders,
                          number_orders_start, simulation_start, allocation_rule, random_state, worker, due_date_range):
    """
    Initialize simulation objects before simulation
    :param priority_rule_start: Start priority dispatching rule for warmup period
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machine types for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param average_count_new_orders: Average count of new orders every week
    :param number_orders_start: Average number of orders at simulation start
    :param simulation_start: Simulation start in clock time of real world
    :param allocation_rule:  Allocation rule for dispatching decision
    :param random_state: Random state for reproducibility
    :param worker: Number of available worker
    :param due_date_range: Due date range for order generation
    :returns: Initialized simpy environment and plant object
    """
    env = simpy.Environment()
    plant = Plant(env, product_types_df, machines_df,
                  orders_df, average_count_new_orders, number_orders_start,
                  simulation_start,
                  priority_rule_start, allocation_rule, random_state, worker,
                  due_date_range)
    env.run(24 * 3600)
    plant.calculate_metrics()
    plant.calculate_state()
    return env, plant


def run_simulation_period(env, plant, model, priority_rules, approach, random_state, classification=True):
    """
    Create machine objects based on information
    :param env: Simpy environment in which the machine is used
    :param plant: Plant object for simulation
    :param model: Model for selection of priority dispatching rules
    :param priority_rules: List of priority rules to choose from for the model
    :param approach: Description of the approach
    :param random_state: Random state for reproducibility
    :param classification: Flag if it is classification approach or regression approach
    :returns: List of resulting metrics of the simulation
    """
    res_list = []
    for day in range(30):
        df = pd.DataFrame(plant.state, index=[0])
        df["priority_rule_start"] = plant.priority_rule.__name__
        if classification:
            rule = model.predict(df)[0]
            plant.priority_rule = priority_rules[rule]
        else:
            rules = []
            score_list = []
            for priority_rule in priority_rules.keys():
                df["priority_rule_score"] = priority_rule
                rules.append(priority_rules[priority_rule])
                score_list.append(model.predict(df))
            plant.priority_rule = rules[np.argmax(score_list)]
        env.run(env.now + 24 * 3600)
        plant.calculate_metrics()
        plant.calculate_state()
        res_list.append({"priority_rule_score": plant.priority_rule.__name__,
                         "day": day,
                         "worker": plant.worker,
                         "due_date_range": plant.due_date_range,
                         "number_orders_start": plant.number_orders_start,
                         "average_count_new_orders": plant.average_count_new_orders,
                         "random_seed": random_state,
                         "revenue": plant.revenue_today,
                         "penalty": plant.penalty_today,
                         "approach": approach})
    return res_list


def run_simulation_complete(model, priority_rules, random_seeds, due_date_ranges, numbers_orders_start,
                            average_counts_new_orders, workers, approach_name, product_types_df, machines_df, orders_df,
                            simulation_start, classification=True):
    """
    Run simulation to evaluate model
    :param model: Model for selection of priority dispatching rules
    :param priority_rules: List of priority rules to choose from for the model
    :param random_seeds: List of random seeds for simulation
    :param due_date_ranges: List of due date ranges for order generation
    :param numbers_orders_start: Average number of orders to start
    :param average_counts_new_orders: Average number of orders each week start
    :param workers: Number of worker
    :param approach_name: Approach name
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machine types for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param simulation_start: Simulation start in clock time of real world
    :param classification: Flag if it is classification approach or regression approach
    :returns: Dataframe of resulting metrics of the simulation
    """
    results = []
    for random_seed in tqdm(random_seeds):
        for due_date_range in due_date_ranges:
            for number_orders_start in numbers_orders_start:
                for average_count_new_orders in average_counts_new_orders:
                    for worker in workers:
                        random_seed += 1
                        env, plant = initialize_simulation(assign_priority_edd, product_types_df, machines_df,
                                                           orders_df, average_count_new_orders,
                                                           number_orders_start, simulation_start, select_machine_winq,
                                                           random_seed, worker, due_date_range)
                        res_list = run_simulation_period(env, plant, model, priority_rules, approach_name,
                                                         random_seed, classification)
                        results.extend(res_list)
    return pd.DataFrame(results)
