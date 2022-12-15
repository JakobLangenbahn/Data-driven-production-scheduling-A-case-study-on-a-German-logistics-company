""" Functions to execute factory simulations """
import pickle
import sys
from datetime import datetime
from time import time, mktime
from random import seed
import pandas as pd
import simpy
from tqdm import tqdm
from src.simulation import Plant, assign_priority_edd, assign_priority_mdd, assign_priority_srpt, \
    assign_priority_spt, assign_priority_lpt, assign_priority_cr, select_machine_winq, assign_priority_ds, \
    assign_priority_fifo


def run_simulation(product_types_df, machines_df, orders_df,
                   number_orders_start, average_count_new_orders,
                   random_state, run_days, simulation_start, priority_rule_start, allocation_rule,
                   priority_rule_score, worker, due_date_range):
    """Function to simulate plant with different rules
    Create machine objects based on information
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machines for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param number_orders_start: Average number of orders at simulation start
    :param average_count_new_orders: Average count of new orders every week
    :param random_state: Random state for reproducibility
    :param run_days: Number of days to run the simulation
    :param simulation_start: Simulation start in clock time of real world
    :param priority_rule_start: Start priority dispatching rule for warmup period
    :param allocation_rule: Start allocation dispatching rule for warmup period
    :param priority_rule_score: Scoring priority dispatching rule for warmup period
    :param worker: Number of available worker
    :returns: Simulated plant objects and simulation time
    """
    start = time()
    seed(random_state)

    env = simpy.Environment()

    simulation_duration = (run_days * 24) * 3600

    plant = Plant(env, product_types_df, machines_df,
                  orders_df, average_count_new_orders, number_orders_start, simulation_start,
                  priority_rule_start, allocation_rule, random_state, worker, due_date_range)

    # Run simulation until decision time
    env.run(until=simulation_duration - 48 * 3600)
    plant.priority_rule = priority_rule_score
    env.run(until=simulation_duration - 24 * 3600)
    plant.calculate_state()
    env.run(until=simulation_duration)
    plant.calculate_metrics()
    end = time()

    return plant, end - start


# Code to run simulation
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing random seed input.")
    elif len(sys.argv) > 2:
        raise ValueError("Too many input variables.")
    else:
        if type(int(sys.argv[1])) != int:
            print(type(sys.argv[1]))
            raise ValueError("Random seed must be an integer.")

    print(f"Start simulation with random seed {sys.argv[1]}")
    random_state = int(sys.argv[1])
    SIMULATION_START = mktime(datetime(2022, 11, 14, 5, 0, 0).timetuple()) * 1000
    MTTF = 60 * 60 * 24 * 7

    product_types_df = pd.read_csv("data/external/product_types.csv")
    with open(r"data/interim/sim_data.pickle", "rb") as output_file:
        orders_df = pickle.load(output_file)
    machines_df = pd.read_csv("data/external/machine.csv")
    machines_df = machines_df[machines_df.product_type_id != 2]

    priority_rules = [assign_priority_edd, assign_priority_mdd,
                      assign_priority_spt, assign_priority_srpt,
                      assign_priority_lpt, assign_priority_fifo,
                      assign_priority_cr, assign_priority_ds]

    res_list = []
    for due_date_range in tqdm([(3, 10), (5, 14), (7, 21)]):
        for number_orders_start in tqdm([80, 90, 100, 110]):
            for average_count_new_orders in [80, 90, 100, 110]:
                for worker in [40, 50, 60, 70]:
                    for day in [3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]:
                        # Change random state before each simulation to increase variety of data
                        random_state += 1
                        for rule_start in priority_rules:
                            for rule_score in priority_rules:
                                plant, duration = run_simulation(product_types_df, machines_df, orders_df,
                                                                 number_orders_start=number_orders_start,
                                                                 average_count_new_orders=average_count_new_orders,
                                                                 random_state=random_state, run_days=day,
                                                                 simulation_start=SIMULATION_START,
                                                                 priority_rule_start=rule_start,
                                                                 allocation_rule=select_machine_winq,
                                                                 priority_rule_score=rule_score,
                                                                 worker=worker,
                                                                 due_date_range=due_date_range)

                                res_list.append({"priority_rule_start": rule_start.__name__,
                                                 "priority_rule_score": rule_score.__name__,
                                                 "day": day,
                                                 "state": plant.state,
                                                 "worker": worker,
                                                 "duration": duration,
                                                 "due_date_range": due_date_range,
                                                 "number_orders_start": number_orders_start,
                                                 "average_count_new_orders": average_count_new_orders,
                                                 "random_seed": random_state,
                                                 "revenue": plant.revenue_today,
                                                 "penalty": plant.penalty_today})
    res_df = pd.DataFrame(res_list)
    with open(r"data/interim/results_simulation.pickle", "wb") as output_file:
        pickle.dump(res_df, output_file)
