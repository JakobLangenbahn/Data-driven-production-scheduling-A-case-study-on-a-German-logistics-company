""" Define priority and allocation dispatching rules """
import pandas as pd
from datetime import datetime
from src.utils import transform_to_real_time


def assign_priority_edd(orders):
    """
    Assign priorities to orders based on earliest due date heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        order_list.append({"Order": order,
                           "dt_due": order.due_date,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["dt_due", "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_mdd(orders):
    """
    Assign priorities to orders based on modified due date heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        plant = order.plant
        minimal_date = datetime.fromtimestamp(
            transform_to_real_time(order.predicted_processing_time_total + plant.env.now,
                                   plant.simulation_start) / 1000)
        order_list.append({"Order": order,
                           "dt_due": max(minimal_date, order.due_date),
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["dt_due", "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_spt(orders):
    """
    Assign priorities to orders based on shortest processing times heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        order_list.append({"Order": order,
                           "predicted_processing_time": order.predicted_processing_time_total,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["predicted_processing_time",
                                             "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_srpt(orders):
    """
    Assign priorities to orders based on shortest remaining processing times heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        predicted_remaining_processing_time = order.predicted_processing_time_unit * (
                order.number_of_pallets - order.output_count_order)
        order_list.append({"Order": order,
                           "predicted_remaining_processing_time": predicted_remaining_processing_time,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["predicted_remaining_processing_time",
                                             "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_lpt(orders):
    """
    Assign priorities to orders based on longest processing times heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        order_list.append({"Order": order,
                           "predicted_processing_time": order.predicted_processing_time_total,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["predicted_processing_time",
                                             "pallets_planned"], ascending=[False, True]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_fifo(orders):
    """
    Assign priorities to orders based on first in first out heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        order_list.append({"Order": order,
                           "index": order.index,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["index",
                                             "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_cr(orders):
    """
    Assign priorities to orders based on critical ration heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        plant = order.plant
        date_today = datetime.fromtimestamp(transform_to_real_time(plant.env.now, plant.simulation_start) / 1000)
        due_date_tightness = (order.due_date - date_today).seconds
        order_list.append({"Order": order,
                           "critical_ratio": due_date_tightness / order.predicted_processing_time_total,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["critical_ratio",
                                             "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def assign_priority_ds(orders):
    """
    Assign priorities to orders based on minimal slack heuristic
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    for order in orders:
        plant = order.plant
        date_today = datetime.fromtimestamp(transform_to_real_time(plant.env.now, plant.simulation_start) / 1000)
        due_date_tightness = (order.due_date - date_today).seconds - order.predicted_processing_time_total
        order_list.append({"Order": order,
                           "critical_ratio": due_date_tightness - order.predicted_processing_time_total,
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        order_subset = order_df.sort_values(["critical_ratio",
                                             "pallets_planned"]).reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def select_machine_ninq(order, machines):
    """
    Select machine for order based on minimal number of orders in queue dispatching rule
    :param order: Order object which has to be assigned to a machine
    :param machines: Set of machine objects which are available for assignment
    :returns: Selected machine for order
    """
    selected_machine = None
    if not order.machine:
        current_shortest_queue = 10000
        available_machines = [machine for machine in machines if
                              machine.product_type == order.product_type and machine.available]
        for machine in available_machines:
            length_queue_machine = len(machine.queue)
            if length_queue_machine < current_shortest_queue:
                current_shortest_queue = length_queue_machine
                selected_machine = machine
    else:
        selected_machine = order.machine
    return selected_machine


def select_machine_winq(order, machines):
    """
    Select machine for order based on minimal amount of worker in queue dispatching rule
    :param order: Order object which has to be assigned to a machine
    :param machines: Set of machine objects which are available for assignment
    :returns: Selected machine for order
    """
    selected_machine = None
    if not order.machine:
        current_shortest_buffer = 1000000000000
        available_machines = [machine for machine in machines if
                              machine.product_type == order.product_type and machine.available]
        if available_machines:
            for machine in available_machines:
                if machine.queue:
                    buffer_time = sum([queue_order.predicted_processing_time_unit * (
                            queue_order.number_of_pallets - queue_order.output_count_order) for queue_order in
                                       machine.queue])
                else:
                    buffer_time = 0
                if buffer_time < current_shortest_buffer:
                    current_shortest_buffer = buffer_time
                    selected_machine = machine
    else:
        selected_machine = order.machine
    return selected_machine


def composite_priority_dispatching_rule(orders):
    """
    Assign priorities to orders based on a composite priority dispatching rule build from edd, cr, spt, srpt
    :param orders: Set of order objects to assign priorities to
    """
    order_list = []
    plant = orders[0].plant
    for order in orders:
        predicted_remaining_processing_time = order.predicted_processing_time_unit * (
                order.number_of_pallets - order.output_count_order)
        date_today = datetime.fromtimestamp(
            transform_to_real_time(order.plant.env.now, order.plant.simulation_start) / 1000)
        due_date_tightness = (order.due_date - date_today).seconds
        order_list.append({"Order": order,
                           "predicted_processing_time": order.predicted_processing_time_total,
                           "predicted_remaining_processing_time": predicted_remaining_processing_time,
                           "dt_due": order.due_date,
                           "critical_ratio": due_date_tightness / min(order.predicted_processing_time_total, 1),
                           "pallets_planned": order.number_of_pallets})
    order_df = pd.DataFrame(order_list)
    if not order_df.empty:
        priority_edd = plant.weights[0] * order_df.sort_values(["dt_due", "pallets_planned"]).index
        priority_spt = plant.weights[1] * order_df.sort_values(["predicted_processing_time", "pallets_planned"]).index
        priority_srpt = plant.weights[2] * order_df.sort_values(
            ["predicted_remaining_processing_time", "pallets_planned"]).index
        priority_cr = plant.weights[3] * order_df.sort_values(["critical_ratio", "pallets_planned"]).index
        order_df["priority_composite"] = priority_edd + priority_spt + priority_srpt + priority_cr
        order_subset = order_df.sort_values("priority_composite").reset_index(drop=True)
        for num, row in order_subset.iterrows():
            row.Order.priority = num


def composite_allocation_dispatching_rule(order, machines):
    """
    Select machine for order based on a composite allocation dispatching rule build from ninq and winq
    :param order: Order object which has to be assigned to a machine
    :param machines: Set of machine objects which are available for assignment
    :returns: Selected machine for order
    """
    selected_machine = None
    plant = order.plant
    if not order.machine:
        available_machines = [machine for machine in machines if
                              machine.product_type == order.product_type and machine.available]
        machine_list = []
        for machine in available_machines:
            length_queue_machine = len(machine.queue)
            if machine.queue:
                buffer_time = sum([queue_order.predicted_processing_time_total * (
                        queue_order.number_of_pallets - queue_order.output_count_order) for queue_order in
                                   machine.queue])
            else:
                buffer_time = 0
            machine_list.append({"Machine": machine,
                                 "Buffer Time": buffer_time,
                                 "Queue length": length_queue_machine})
        machine_df = pd.DataFrame(machine_list)
        if not machine_df.empty:
            priority_ninq = plant.weights[0][4] * machine_df.sort_values("Queue length").index
            priority_winq = plant.weights[0][5] * machine_df.sort_values("Buffer Time").index
            machine_df["priority_composite"] = priority_ninq + priority_winq
            machine_subset = machine_df.sort_values("priority_composite").reset_index(drop=True)
            selected_machine = machine_subset.Machine[0]
    else:
        selected_machine = order.machine
    return selected_machine
