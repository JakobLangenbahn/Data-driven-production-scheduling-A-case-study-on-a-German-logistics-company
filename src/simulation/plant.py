""" Define class for plant for catering production processes """
from datetime import datetime, timedelta
from math import floor
from random import randrange, seed, sample, choice
from statistics import mean, median, variance
import simpy
from src.simulation import Machine, ProductType, Order
from src.utils import transform_to_real_time, random_range


class Plant(object):
    """
    Class for saving product type information
    :param env: Simpy environment in which the machine is used
    :param product_types_df: Dataframe with information about product types for product type generation
    :param machines_df: Dataframe with information about machines for machine object generation
    :param orders_df: Dataframe with information about orders for order object generation
    :param average_count_new_orders: Average count of new orders every week
    :param number_orders_start: Average number of orders at simulation start
    :param simulation_start: Simulation start in clock time of real world
    :param priority_rule: Priority rule for dispatching decision
    :param allocation_rule: Allocation rule for dispatching decision
    :param random_state: Random state for reproducibility
    :param worker: Number of available worker
    :param due_date_range: Due date range for order generation
    :param dispatching: Boolean variable to mark if approach is selecting dispatching rule or assigning jobs directly
    """

    def __init__(self, env, product_types_df,
                 machines_df, orders_df, average_count_new_orders,
                 number_orders_start, simulation_start, priority_rule, allocation_rule,
                 random_state, worker, due_date_range, dispatching=True):
        """
        Constructor method
        """
        # Initialize simulation environment
        self.env = env
        self.random_state = random_state
        self.simulation_start = simulation_start
        self.step_event = self.env.event()
        self.dispatching = dispatching
        # Initialize szenario attributes
        self.number_orders_start = number_orders_start
        self.average_count_new_orders = average_count_new_orders
        self.worker = worker
        self.due_date_range = due_date_range
        # Initialize dispatching rules
        self.allocation_rule = allocation_rule
        self.priority_rule = priority_rule
        # Initialize plant resources and orders
        self.orders_df = orders_df
        self.product_types = self.create_product_types(product_types_df)
        self.machines = self.create_machines(machines_df)
        self.open_orders = self.generate_orders(number_orders_start, initial=True)
        self.uninitialized_orders = self.open_orders[:]
        self.worker_shift = simpy.Container(self.env, init=self.worker)
        # Initialize data collection container
        self.state = None
        self.finished_orders = []
        self.unit_sessions = []
        self.production_sessions = []
        self.unit_sessions_today = []
        self.states = []
        # Initialize dynamic attributes
        self.revenue_today = 0
        self.penalty_today = 0
        self.revenue_total = 0
        self.penalty_total = 0
        self.penalty_step = 0
        self.revenue_step = 0
        self.weights = [0.5, 0.5, 0.5, 0.5]

        # Start shift processes
        env.process(self.shift_management())

    @staticmethod
    def create_product_types(product_types_df):
        """
        Create product type objects based on information
        :param product_types_df: Dataframe with information about product types for product type object generation
        :returns: Product type objects
        """
        return [ProductType(product_type_id=row.id,
                            name=row.product_type,
                            setup_time=row.setup_time_hours * 3600,
                            number_of_worker=row.workers_per_station) for index, row in product_types_df.iterrows()]

    def create_machines(self, machines_df):
        """
        Create machine objects based on information
        :param machines_df: Dataframe with information about machine types for machine object generation
        :returns: Machine objects
        """
        return [Machine(env=self.env,
                        machine_id=row.id,
                        product_type=[product_type for product_type in self.product_types if
                                      product_type.product_type_id == row.product_type_id][0]) for index, row in
                machines_df.iterrows()]

    def generate_orders(self, average_number_of_orders, initial):
        """
        Calculate break time based on cause
        :param average_number_of_orders: Average number of orders to generate
        :param initial: Flag if it is the generation of the initial order or orders at the beginning of the week
        :returns: List of order objects
        """
        # We want to keep the distribution of products and product types
        # Therefore we draw from the real world order set
        week = floor(self.env.now / (7 * 24 * 3600))
        number_of_orders = max(average_number_of_orders + random_range(-10, 10, int(self.random_state * week)), 0)
        if not initial:
            number_existing_orders = len(self.open_orders) + len(self.finished_orders)
        else:
            number_existing_orders = 0
        order_subset = self.orders_df.sample(number_of_orders, random_state=int(self.random_state * week)).reset_index(
            drop=True)
        date_today = datetime.fromtimestamp(transform_to_real_time(self.env.now, self.simulation_start) / 1000)
        return [Order(self.env,
                      self,
                      f"Order_{number_existing_orders + index}",
                      [product_type for product_type in self.product_types if
                       product_type.product_type_id == row.product_type_id][0],
                      max(row.pallets_planned + random_range(-round(row.pallets_planned * 0.4),
                                                             round(row.pallets_planned * 0.4) + 1,
                                                             self.random_state + index), 1),
                      # Set values based on data of plant
                      date_today + timedelta(
                          random_range(self.due_date_range[0], self.due_date_range[1], self.random_state + index)),
                      row.product_id,
                      row.boxes_per_pallet,
                      row.unit_cost_product,
                      number_existing_orders + index,
                      row.distr) for index, row in order_subset.iterrows()]

    def clean_machines(self, end_of_day=False):
        """
        Clear machines from orders
        :param end_of_day: Flag it is the end of the day
        """
        for machine in self.machines:
            machine.queue = []
            machine.resource = None
            machine.resource = simpy.PriorityResource(self.env, capacity=1)
            if end_of_day:
                machine.time_produced_today = 0

    def clean_worker(self):
        """
        Clear worker restrictions after shift end
        """
        self.worker_shift = None
        self.worker_shift = simpy.Container(self.env, init=self.worker)

    def end_shift(self, shift_name):
        """
        End shift and prepare next shift
        :param shift_name: Name of the finished shift
        """
        for order in self.open_orders:
            if order.process and not order.process.triggered and not order.blocked:
                order.process.interrupt(cause=f"{shift_name} ends")
        self.clean_machines(end_of_day=True)
        self.clean_worker()
        self.order_change()
        if shift_name == "Evening Shift Weekend":
            new_orders = self.generate_orders(self.average_count_new_orders, False)
            self.open_orders.extend(new_orders)
            self.uninitialized_orders.extend(new_orders)
            # Initialize for the case that no orders where left in the end of last week
            self.step_event.succeed()
            self.step_event = self.env.event()
        if self.dispatching:
            self.priority_rule(self.open_orders)
        else:
            date_today = datetime.fromtimestamp(
                transform_to_real_time(self.env.now, self.simulation_start) / 1000)
            if self.open_orders:
                for order in self.open_orders:
                    # Smaller or equal as it takes one day to deliver the order
                    if order.due_date <= date_today:
                        # Only complete delivered get no penalty
                        self.penalty_step += order.unit_cost * order.boxes_per_pallet * order.number_of_pallets * 0.05

    def order_change(self, amount_to_change=2):
        """
        Function to randomly change order attributes as for
        example due date or pallets planned may change
        :param amount_to_change: Amount of orders to maximal change
        """
        day = floor(self.env.now / (24 * 3600))
        seed(self.random_state + day)
        number_of_orders = randrange(amount_to_change + 1)
        if self.open_orders and len(self.open_orders) >= number_of_orders:
            orders_to_change = sample(self.open_orders, number_of_orders)
            if orders_to_change:
                for order in orders_to_change:
                    if choice([True, False]):
                        order.number_of_pallets = max(
                            order.number_of_pallets + randrange(-round(order.number_of_pallets * 0.2),
                                                                round(order.number_of_pallets * 0.2) + 1), 1)
                    else:
                        date_today = datetime.fromtimestamp(
                            transform_to_real_time(self.env.now, self.simulation_start) / 1000)
                        order.due_date = date_today + timedelta(randrange(1, 4))

    def calculate_metrics(self):
        """
        Calculate metrics based on system state
        """
        # Calculate revenue of produced orders
        self.revenue_today = 0
        for unit in self.unit_sessions_today:
            self.revenue_today += unit["Revenue"]
            self.revenue_total += self.revenue_today

        # Calculate penalty for late orders
        date_today = datetime.fromtimestamp(transform_to_real_time(self.env.now, self.simulation_start) / 1000)
        self.penalty_today = 0
        if self.open_orders:
            for order in self.open_orders:
                # Smaller or equal as it takes one day to deliver the order
                if order.due_date <= date_today:
                    self.penalty_today += order.unit_cost * order.boxes_per_pallet * order.number_of_pallets * 0.05
                    self.penalty_total += self.penalty_today

    def calculate_state(self):
        """
        Calculate system state of the plant
        """
        state_dict = {}

        # Prepare required calculations
        date_today = datetime.fromtimestamp(transform_to_real_time(self.env.now, self.simulation_start) / 1000)
        if self.open_orders:
            for order in self.open_orders:
                order.remaining_processing_time = order.predicted_processing_time_unit * (
                        order.number_of_pallets - order.output_count_order)
                order.due_date_tightness = (order.due_date - date_today).seconds
                order.slack_time = (order.due_date - date_today).days * 19 * 3600 - order.remaining_processing_time

        state_dict["number_available_machines"] = len([machine for machine in self.machines if machine.available])
        state_dict["number_of_jobs"] = len(self.open_orders)
        for product_type in self.product_types:
            state_dict[f"number_available_machines_{product_type.product_type_id}"] = len(
                [machine for machine in self.machines if
                 machine.product_type == product_type and machine.available])
            orders_product_type = [order for order in self.open_orders if order.product_type == product_type]
            state_dict[f"number_of_jobs_{product_type.product_type_id}"] = len(orders_product_type)
            state_dict[f"remaining_processing_time_{product_type.product_type_id}"] = sum(
                [order.remaining_processing_time for order in orders_product_type])

        if self.open_orders and len(self.open_orders) > 1:
            processing_times = [order.predicted_processing_time_total for order in self.open_orders]
            remaining_processing_times = [order.remaining_processing_time for order in self.open_orders]
            due_date_tightness = [order.due_date_tightness for order in self.open_orders]
            slack_times = [order.slack_time for order in self.open_orders]
            sojourn_times = [(date_today - order.arrival_date).days for order in self.open_orders]
            completion_rates = [(order.output_count_order / order.number_of_pallets) for order
                                in self.open_orders]

            for statistic_function in [max, min, mean, median, variance, sum]:
                state_dict[f"processing_time_{statistic_function.__name__}"] = statistic_function(processing_times)
                state_dict[f"remaining_processing_time_{statistic_function.__name__}"] = statistic_function(
                    remaining_processing_times)
                state_dict[f"slack_time_{statistic_function.__name__}"] = statistic_function(slack_times)
                state_dict[f"due_date_tightness_{statistic_function.__name__}"] = statistic_function(due_date_tightness)
                state_dict[f"sojourn_time_{statistic_function.__name__}"] = statistic_function(sojourn_times)
                state_dict[f"completion_rate_{statistic_function.__name__}"] = statistic_function(completion_rates)
        else:
            for statistic_function in [max, min, mean, median, variance, sum]:
                state_dict[f"processing_time_{statistic_function.__name__}"] = 0
                state_dict[f"remaining_processing_time_{statistic_function.__name__}"] = 0
                state_dict[f"slack_time_{statistic_function.__name__}"] = 0
                state_dict[f"due_date_tightness_{statistic_function.__name__}"] = 0
                state_dict[f"sojourn_time_{statistic_function.__name__}"] = 0
                state_dict[f"completion_rate_{statistic_function.__name__}"] = 0

        # Calculate metrics
        self.calculate_metrics()
        state_dict["revenue_today"] = self.revenue_today
        state_dict["penalty_today"] = self.penalty_today

        # Assign to product
        self.state = state_dict

    def shift_management(self):
        """
        Shift management process to simulate shifts and start production
        """
        if self.dispatching:
            self.priority_rule(self.open_orders)
            for order in self.open_orders:
                order.initialize()
        while True:
            time = (self.env.now + (5 * 3600)) % (24 * 3600)
            day = (self.env.now + (5 * 3600)) % (24 * 3600 * 7)
            if time == 5 * 3600:
                self.unit_sessions_today = []
                yield self.env.timeout(10 * 60 * 60)
            elif time == 15 * 3600 and day <= (3 * 24 + 15) * 3600:
                yield self.env.timeout(9 * 3600)
                self.end_shift("Evening Shift")
                yield self.env.timeout(5 * 3600)
            elif time == 15 * 3600 and day == (4 * 24 + 15) * 3600:
                yield self.env.timeout(9 * 3600)
                self.end_shift("Evening Shift Weekend")
                yield self.env.timeout((5 + 24 + 24) * 60 * 60)
                if self.dispatching:
                    for order in self.open_orders:
                        if not order.process:
                            order.initialize()
