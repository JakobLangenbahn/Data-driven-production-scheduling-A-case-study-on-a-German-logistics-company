"""Code of the implementation of an automatic scheduler at a German logistic company
The code is not thought to be run, as it was developed in another environment requiring Python 3.6.
It should merely showcase and explain the final implemented algorithm.
"""
from datetime import timedelta, date
from math import ceil

import pandas as pd

# Imports from not included flask app
from app import db
from app.models import Station, Order, Product, ProductType, ScheduleItem
from app.utils import db_safe_commit


def create_query_orders(plant_id):
    """
    Function to create a query for orders and related information used at different points of the scheduler
    :param plant_id: An unique identifier of the plant in the database
    :returns: A query to the database to get relevant information for scheduling
    """
    query = db.session.query(Order, Product, ProductType)
    query = query.join(Product, Order.product_id == Product.id, isouter=True)
    query = query.join(ProductType, ProductType.id == Product.product_type_id, isouter=True)
    query = query.filter(Order.plant_id == plant_id)
    query = query.filter(Order.stage.in_((0, 1, 2)))
    # Remove all orders that do not contain the required data, which is nullable in the database
    query = query.filter(Order.dt_due != None)
    query = query.filter(Product.product_type_id != None)
    query = query.filter(Product.boxes_per_worker_per_hour != None)
    query = query.filter(Product.boxes_per_pallet != None)
    query = query.filter(ProductType.workers_per_station != None)
    query = query.filter(ProductType.setup_time_hours != None)
    # Remove all orders that do not have conclusive data
    query = query.filter(ProductType.workers_per_station > 0)
    query = query.filter(Product.boxes_per_pallet > 0)
    query = query.filter(Product.boxes_per_worker_per_hour > 0)
    return query


def query_calendar_weeks_to_plan(plant_id):
    """
    Function to query the calendar weeks to be planned by the scheduling algorithm based on due dates
    :param plant_id: An unique identifier of the plant in the database
    :returns: Lists including which days and weeks the scheduler has to include
    """
    query = create_query_orders(plant_id)
    order_with_latest_due_date = query.order_by(Order.dt_due.desc()).first()
    last_date = order_with_latest_due_date[0].dt_due
    today = date.today()
    delta = last_date - today
    # Add 7 more days to the range to make sure that the whole calendar week of the last due date is planned
    # Filter out days that are on the weekend, assuming that there the algorithm does not plan shifts on weekends
    days_for_scheduling = [today + timedelta(days=days) for days in range(max(delta.days, 28)) if
                           (today + timedelta(days=days)).isoweekday() not in (6, 7)]
    weeks_for_scheduling = list(dict.fromkeys([day.isocalendar()[0:2] for day in days_for_scheduling]))
    return days_for_scheduling, weeks_for_scheduling


def calculate_worker_per_shift(days_for_scheduling, weeks_for_scheduling, plant_id, number_of_days=5, number_of_hours=9,
                               maximal_worker_shift=70, minimal_worker_shift_late=20, minimal_worker_shift_early=50):
    """
    Function to estimate the number of workers needed for the early and late shifts for specified calendar weeks
    :param days_for_scheduling: List of days for scheduling
    :param weeks_for_scheduling: List of weeks for scheduling
    :param plant_id: An unique identifier of the plant in the database
    :param number_of_days: Number of workdays in a week
    :param number_of_hours: Number of hours in a shift
    :param maximal_worker_shift: Maximum number of workers in a shift
    :param minimal_worker_shift_late: Minimal number of workers in the late shift
    :param minimal_worker_shift_early: Minimal number of workers in the early shift
    :returns: A list including the planned number of workers for each day for scheduling
    """
    capacity_day = {}
    manhours_carry_over = 0
    for week in weeks_for_scheduling:
        manhours_week = 0 + manhours_carry_over
        if week == date.today().isocalendar()[0:2]:
            # When planning the current week also include orders with due dates in the past
            days = [day for day in days_for_scheduling if day.isocalendar()[0:2] == week]
            query = create_query_orders(plant_id)
            query = query.filter(Order.dt_due < max(days))
            order_list = query.all()
        else:
            days = [day for day in days_for_scheduling if day.isocalendar()[0:2] == week]
            query = create_query_orders(plant_id)
            query = query.filter(Order.dt_due.in_(days))
            order_list = query.all()
        for order in order_list:
            # Estimate remaining manhours based on the expected amount of boxes per worker per hour and the remaining
            # amount of pallets to be produced
            estimated_remaining_manhours = (
                    (order[0].boxes_planned - order[0].current_output_count * order[1].boxes_per_pallet) / order[
                1].boxes_per_worker_per_hour)
            manhours_week = manhours_week + estimated_remaining_manhours
        worker_required_shifts = manhours_week / (number_of_days * number_of_hours)
        # If possible, then produce everything in the morning shift, else produce as much as possible in the morning
        # shift and produce the remains in the late shift but make sure that the minimum required amount of workers per
        # shift is reached
        if worker_required_shifts > maximal_worker_shift:
            if worker_required_shifts > minimal_worker_shift_late + maximal_worker_shift:
                if worker_required_shifts > maximal_worker_shift + maximal_worker_shift:
                    worker_early_shift = maximal_worker_shift
                    worker_late_shift = maximal_worker_shift
                    manhours_carry_over = (worker_required_shifts - 2 * maximal_worker_shift) * number_of_hours
                else:
                    worker_early_shift = maximal_worker_shift
                    worker_late_shift = ceil(worker_required_shifts) - maximal_worker_shift
            else:
                worker_early_shift = max(ceil(worker_required_shifts) - minimal_worker_shift_late,
                                         minimal_worker_shift_early)
                worker_late_shift = minimal_worker_shift_late
        else:
            worker_early_shift = max([ceil(worker_required_shifts), minimal_worker_shift_early])
            worker_late_shift = 0
        for day in days:
            capacity_day[day] = [worker_early_shift, worker_late_shift]
    return capacity_day


def create_capacity_restrictions(capacity_day, plant_id):
    """
    Function to create a capacity restrictions dataframe used in the scheduling algorithm
    :param capacity_day: A list including the planned number of workers for each day for scheduling
    :param plant_id: An unique identifier of the plant in the database
    :returns: Dataframes for global shift capacities and station shift capacities
    """
    query = db.session.query(Station)
    query = query.filter(Station.plant_id == plant_id)
    # Orders cannot be assigned to indirect stations or stations with missing product types
    query = query.filter(Station.indirect == 0)
    query = query.filter(Station.product_type_id != None)
    stations = query.all()
    stations_df = pd.DataFrame({"station_id": [station.id for station in stations],
                                "product_type_id": [station.product_type_id for station in stations]})
    # Add 1 and 2 as shift ids for the columns
    days = pd.DataFrame.from_dict(capacity_day, orient="index", columns=["1", "2"])
    days_new = days.reset_index()
    days_new.columns = ["dt", "1", "2"]
    capacity_shifts = days_new.melt(id_vars=["dt"], value_vars=['1', "2"], var_name="shift_id",
                                    value_name="capacity_worker")
    capacity_shifts["calender_week"] = pd.to_datetime(capacity_shifts["dt"]).dt.week
    capacity_shifts["capacity_manhours"] = capacity_shifts["capacity_worker"] * 9
    capacity_shifts.drop(["capacity_worker"], axis=1, inplace=True)
    capacity_shifts["merge_index"] = 1
    stations_df["merge_index"] = 1
    capacity_shifts_stations = pd.merge(capacity_shifts, stations_df, on='merge_index')
    capacity_shifts_stations.drop(["merge_index"], axis=1, inplace=True)
    capacity_shifts_stations["hours_working_clock"] = 9
    capacity_shifts_stations["amount_of_orders"] = 0
    capacity_shifts_stations.drop(["capacity_manhours"], axis=1, inplace=True)
    return capacity_shifts, capacity_shifts_stations


def find_available_stations_shifts_for_order(order, capacity_shifts, capacity_shifts_stations,
                                             minimal_amount_clock_hours_work=1):
    """
    Function to find all available station shifts combination for the scheduling of a specific order
    :param order: An order to be scheduled
    :param capacity_shifts: Dataframe for global shift capacities
    :param capacity_shifts_stations: Dataframe for station shift capacities
    :param minimal_amount_clock_hours_work: Minimal amount of clock time remaining in shift to start a new order
    :returns: A list of potential shifts for scheduling the order
    """
    available_stations_shifts = capacity_shifts_stations[
        capacity_shifts_stations.product_type_id == order[1].product_type_id]
    # Check if the station has enough capacity in this shift based on some manually set minimal threshold
    available_stations_shifts = available_stations_shifts[
        available_stations_shifts.hours_working_clock > minimal_amount_clock_hours_work]
    # Check if the shift has enough capacity for the order
    available_shifts = capacity_shifts[
        capacity_shifts.capacity_manhours > order[2].workers_per_station * minimal_amount_clock_hours_work]
    keys = ["dt", "shift_id"]
    i1 = available_stations_shifts.set_index(keys).index
    i2 = available_shifts.set_index(keys).index
    available_stations_shifts = available_stations_shifts[i1.isin(i2)]
    return available_stations_shifts.sort_values(["dt", "shift_id", "hours_working_clock"],
                                                 ascending=[True, True, False])


def query_earliest_available_stations_for_order(available_stations_shifts, order):
    """
    Function to detect the earliest available station shift combination for the order
    :param available_stations_shifts: A list of potential shifts for scheduling the order
    :param order: An order to be scheduled
    :returns: Shifts to schedule the order and meta-information about the planned scheduling item
    """
    first_option = available_stations_shifts.iloc[0, :]
    order_duration = ((order[0].boxes_planned - order[0].current_output_count * order[1].boxes_per_pallet) / (
            order[1].boxes_per_worker_per_hour * order[2].workers_per_station))
    # If the station can be set up for the order and the order is produced in the remaining time of the shift, then assign
    # the order to the shift, else assign it to multiple subsequent possible shifts on the same station.
    if first_option.hours_working_clock > order[2].setup_time_hours + order_duration:
        number_of_shifts = 1
        information = {"dt": first_option["dt"],
                       "shift_id": first_option["shift_id"],
                       "station_id": first_option["station_id"],
                       "amount_of_orders": first_option["amount_of_orders"],
                       "hours_working_clock": first_option["hours_working_clock"]}
    else:
        # We can assume that the following shifts have no order assigned to the station based on the way we query and
        # assign orders
        number_of_shifts = ceil((order[2].setup_time_hours + order_duration - first_option.hours_working_clock) / 9) + 1
        available_stations_shifts_order = available_stations_shifts[
            available_stations_shifts.station_id == first_option.station_id]
        station_shifts = available_stations_shifts_order.iloc[0:number_of_shifts, :]
        information = {"dt": station_shifts["dt"].tolist(),
                       "shift_id": station_shifts["shift_id"].tolist(),
                       "station_id": station_shifts["station_id"].tolist(),
                       "amount_of_orders": station_shifts["amount_of_orders"].tolist(),
                       "hours_working_clock": station_shifts["hours_working_clock"].tolist()}
    return number_of_shifts, information


def assign_orders_to_one_shift(order, information, capacity_shifts, capacity_shifts_stations):
    """
    Function to add an order to a single shift and a single station
    :param order: An order to be scheduled
    :param information: Meta-information about the scheduled item
    :param capacity_shifts: Dataframe for global shift capacities
    :param capacity_shifts_stations: Dataframe for station shift capacities
    :returns: Updated capacities for shifts and stations
    """
    # Add schedule item to the database
    item = ScheduleItem(
        station_id=information["station_id"],
        schedule_shift_id=information["shift_id"],
        order_id=order[0].id,
        priority=information['amount_of_orders'],
        dt=information["dt"],
        is_deleted=False,
        total_splits=1,
        pallets_planned=order[0].pallets_planned
    )
    db.session.add(item)
    db_safe_commit(db)
    # Reduce available capacity at station
    new_amount_of_orders = information['amount_of_orders'] + 1
    order_duration = ((order[0].boxes_planned - order[0].current_output_count * order[1].boxes_per_pallet) / (
            order[1].boxes_per_worker_per_hour * order[2].workers_per_station))
    new_hours_working_clock = information["hours_working_clock"] - (order[2].setup_time_hours + order_duration)
    index_station = (capacity_shifts_stations.station_id == information["station_id"]) & (
            capacity_shifts_stations.dt == information["dt"]) & (
                            capacity_shifts_stations.shift_id == information["shift_id"])
    capacity_shifts_stations['hours_working_clock'].mask(index_station, other=new_hours_working_clock, inplace=True)
    capacity_shifts_stations['amount_of_orders'].mask(index_station, other=new_amount_of_orders, inplace=True)
    # Reduce available capacity at shift
    index_shift = (capacity_shifts.dt == information["dt"]) & (capacity_shifts.shift_id == information["shift_id"])
    new_capacity_manhours = capacity_shifts['capacity_manhours'][index_shift] - (
            order_duration * order[2].workers_per_station)
    capacity_shifts['capacity_manhours'].mask(index_shift, other=new_capacity_manhours, inplace=True)
    return capacity_shifts, capacity_shifts_stations


def assign_orders_to_multiple_shifts(order, number_of_shifts, information, capacity_shifts, capacity_shifts_stations):
    """
    Function to assign an order to multiple shifts on the same station
    :param order: An order to be scheduled
    :param number_of_shifts: Number of shifts this order has to be scheduled in
    :param information: Meta-information about the scheduled item
    :param capacity_shifts: Dataframe for global shift capacities
    :param capacity_shifts_stations: Dataframe for station shift capacities
    :returns: Updated capacities for shifts and stations
    """
    order_duration = ((order[0].boxes_planned - order[0].current_output_count * order[1].boxes_per_pallet) / (
            order[1].boxes_per_worker_per_hour * order[2].workers_per_station))
    total_clock_duration = order[2].setup_time_hours + order_duration
    for i in range(number_of_shifts):
        # Calculate the expected amount of pallets produced based on the percentage of hours worked in the shift of
        # the total working time for the first, last, and the middle shifts
        # Calculate the split duration for reducing the shift and station capacity
        if i == 0:
            expected_number_of_pallets = round(
                ((information["hours_working_clock"][i] - order[2].setup_time_hours) / order_duration) * order[
                    0].pallets_planned)
            split_duration = information["hours_working_clock"][i] - order[2].setup_time_hours
        elif i == (number_of_shifts - 1):
            expected_number_of_pallets = round((total_clock_duration / order_duration) * order[0].pallets_planned)
            split_duration = total_clock_duration
        else:
            expected_number_of_pallets = round(
                (information["hours_working_clock"][i] / order_duration) * order[0].pallets_planned)
            split_duration = information["hours_working_clock"][i]
            # Add schedule item to the database
        item = ScheduleItem(
            station_id=information["station_id"][i],
            schedule_shift_id=information["shift_id"][i],
            order_id=order[0].id,
            priority=information['amount_of_orders'][i],
            dt=information["dt"][i],
            split_duration=split_duration,
            pallets_planned=expected_number_of_pallets,
            split_number=i + 1,
            total_splits=number_of_shifts,
            is_deleted=False
        )
        db.session.add(item)
        db_safe_commit(db)
        # Reduce available capacity at station
        new_amount_of_orders = information['amount_of_orders'][i] + 1
        if i == 0:
            new_hours_working_clock = information["hours_working_clock"][i] - (
                    split_duration + order[2].setup_time_hours)
        else:
            new_hours_working_clock = information["hours_working_clock"][i] - split_duration
        index_station = (capacity_shifts_stations.station_id == information["station_id"][i]) & (
                capacity_shifts_stations.dt == information["dt"][i]) & (
                                capacity_shifts_stations.shift_id == information["shift_id"][i])
        capacity_shifts_stations['hours_working_clock'].mask(index_station, other=new_hours_working_clock, inplace=True)
        capacity_shifts_stations['amount_of_orders'].mask(index_station, other=new_amount_of_orders, inplace=True)
        total_clock_duration = total_clock_duration - split_duration
        # Reduce available capacity at shift
        index_shift = (capacity_shifts.dt == information["dt"][i]) & (
                capacity_shifts.shift_id == information["shift_id"][i])
        new_capacity_manhours = capacity_shifts['capacity_manhours'][index_shift] - (
                split_duration * order[2].workers_per_station)
        capacity_shifts['capacity_manhours'].mask(index_shift, other=new_capacity_manhours, inplace=True)
    return capacity_shifts, capacity_shifts_stations


def schedule_orders(plant_id):
    """
    Function to schedule open orders for the days from the current day on
    :param plant_id: An unique identifier of the plant in the database
    :returns: Updated capacities for shifts and stations
    """
    days_for_scheduling, weeks_for_scheduling = query_calendar_weeks_to_plan(plant_id)
    capacity_day = calculate_worker_per_shift(days_for_scheduling, weeks_for_scheduling, plant_id, number_of_days=5,
                                              number_of_hours=9, maximal_worker_shift=70, minimal_worker_shift_late=20)
    capacity_shifts, capacity_shifts_stations = create_capacity_restrictions(capacity_day, plant_id)
    query = create_query_orders(plant_id)
    # Schedule the orders based on ascending due dates
    query.order_by(Order.dt_due.asc())
    orders = query.all()
    for order in orders:
        print(order[0].id)
        available_stations_shifts = find_available_stations_shifts_for_order(order, capacity_shifts,
                                                                             capacity_shifts_stations,
                                                                             minimal_amount_clock_hours_work=1)
        number_of_shifts, information = query_earliest_available_stations_for_order(available_stations_shifts, order)
        if number_of_shifts == 1:
            capacity_shifts, capacity_shifts_stations = assign_orders_to_one_shift(order, information, capacity_shifts,
                                                                                   capacity_shifts_stations)
        else:
            capacity_shifts, capacity_shifts_stations = assign_orders_to_multiple_shifts(order, number_of_shifts,
                                                                                         information, capacity_shifts,
                                                                                         capacity_shifts_stations)
    return capacity_shifts, capacity_shifts_stations


def remove_future_scheduled_orders(plant_id):
    """
    Remove all existing scheduling to avoid duplication and allow for rescheduling
    :param plant_id: An unique identifier of the plant in the database
    """
    query_station = db.session.query(Station)
    query_station = query_station.filter(Station.plant_id == plant_id)
    query_station = query_station.filter(Station.indirect == 0)
    query_station = query_station.filter(Station.product_type_id != None)
    stations = query_station.all()
    for station in stations:
        query = db.session.query(ScheduleItem)
        query = query.filter(ScheduleItem.station_id == station.id)
        # Delete all scheduled items from today on
        # query = query.filter(ScheduleItem.dt >= date.today())
        schedule_items = query.all()
        for schedule_item in schedule_items:
            db.session.delete(schedule_item)
        db_safe_commit(db)
