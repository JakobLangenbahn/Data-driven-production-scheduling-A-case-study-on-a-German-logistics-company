""" Define order class for representing orders and the production process """
from datetime import datetime
from math import floor

import simpy

from src.utils import transform_to_real_time, generate_random_deviation


class Order(object):
    """
    Class for saving product type information
    :param env: Simpy environment in which the machine is used
    :param plant: Plant object for producing the order in
    :param name: Identifier of the oder
    :param product_type: Product type class of the order
    :param number_of_pallets: Number of pallets planned for the order
    :param due_date: Due date of the order
    :param product_id: Product id
    :param boxes_per_pallet: Number of boxes per pallet
    :param unit_cost: Unit cost for each box
    :param index: Index of arrival
    :param processing_time_distr: Fitted dweibull distribution of the processing times
    """

    def __init__(self, env, plant, name, product_type, number_of_pallets, due_date, product_id,
                 boxes_per_pallet, unit_cost, index, processing_time_distr):
        """
        Constructor method
        """
        # Initialize environment
        self.env = env
        self.plant = plant
        # Initialize fixed attributes
        self.name = name
        self.product_type = product_type
        self.number_of_pallets = number_of_pallets
        self.product_id = product_id
        self.unit_cost = unit_cost
        self.index = index
        self.arrival_date = datetime.fromtimestamp(
            transform_to_real_time(self.env.now, self.plant.simulation_start) / 1000)
        self.processing_time_distr = processing_time_distr
        self.predicted_processing_time_unit = self.processing_time_distr.model["model"].mean()
        self.predicted_processing_time_total = self.predicted_processing_time_unit * number_of_pallets
        self.processing_time = max(
            self.processing_time_distr.generate(1, random_state=int(self.plant.random_state + self.index), verbose=0)[
                0], 300)
        self.boxes_per_pallet = boxes_per_pallet
        self.total_revenue = self.number_of_pallets * self.boxes_per_pallet * self.unit_cost
        # Initialize dynamic attributes
        self.working = False
        self.finished = False
        self.machine = None
        self.due_date = due_date
        self.setup_time = None
        self.setup_complete = False
        self.output_count_order = 0
        self.output_count_production_session = 0
        self.blocked = False
        self.on_break = False
        self.remaining_processing_time = None
        self.priority = 1
        self.process = None

    def initialize(self):
        """
        Initialize production process
        """
        self.process = self.env.process(self.producing())

    def get_unit_processing_time(self):
        """
        Calculate unit processing time based on given distribution
        """
        # second = self.plant.env.now % 60 return max(self.processing_time_distr.generate(1, random_state=int(
        # self.plant.random_state * second), verbose=0)[0], 300)
        return self.processing_time

    def get_setup_time(self, selected_machine):
        """
        Calculate setup time based on predecessor order
        :param selected_machine: Selected machine object
        """
        setup_time = 0
        hour = floor(self.env.now / (24 * 60))
        try:
            last_order_name = selected_machine.last_order_producing_machine[-1]
            if last_order_name != self.name and selected_machine.setup_time:
                random_variation = generate_random_deviation(-selected_machine.setup_time * 0.2,
                                                             selected_machine.setup_time * 0.2, 0,
                                                             selected_machine.setup_time,
                                                             int(self.plant.random_state + hour))
                setup_time = selected_machine.setup_time + random_variation
        except IndexError:
            # If there is no previous order we are on the start of the simulation,
            # and there is the full setup time required.
            if selected_machine.setup_time:
                random_variation = generate_random_deviation(-selected_machine.setup_time * 0.2,
                                                             selected_machine.setup_time * 0.2, 0,
                                                             selected_machine.setup_time,
                                                             int(self.plant.random_state + hour))
                setup_time = selected_machine.setup_time + random_variation
        return setup_time

    @staticmethod
    def calculate_break_time(cause):
        """
        Calculate break time based on cause
        :param cause: Reason for break
        :returns: Break time in seconds
        """
        time_break = 0
        if cause == "Evening Shift ends":
            time_break = 5 * 3600
        elif cause == "Evening Shift Weekend ends":
            time_break = (24 + 24 + 5) * 3600
        return time_break

    def producing(self):
        """
        Method to simulate the production process
        """
        # For initial prioritization the short timeout is used for prioritization
        try:
            yield self.env.timeout(self.priority)
        except simpy.Interrupt as interuption:
            # Calculate break time and go to break
            break_time = self.calculate_break_time(interuption.cause)
            if break_time:
                self.on_break = True
                yield self.env.timeout(break_time)
                self.on_break = False
        while self.output_count_order < self.number_of_pallets:
            # Another prioritization for the beginn of a new shift
            try:
                # Plus for the case of priority 0 and no machine available
                yield self.env.timeout(self.priority + 1)
            except simpy.Interrupt as interuption:
                # Calculate break time and go to break
                break_time = self.calculate_break_time(interuption.cause)
                if break_time:
                    self.on_break = True
                    yield self.env.timeout(break_time)
                    self.on_break = False
                continue

            # Select machine based on dispatching rule
            selected_machine = self.plant.allocation_rule(self, self.plant.machines)

            if selected_machine:
                with selected_machine.resource.request(priority=self.priority) as machine_request:
                    try:
                        selected_machine.queue.append(self)
                        yield machine_request
                        # Stop if the obtained machine is not available due to failure
                        if not selected_machine.available:
                            selected_machine.current_order = None
                            if self in selected_machine.queue:
                                selected_machine.queue.remove(self)
                            self.machine = None
                            continue
                    except simpy.Interrupt as interuption:
                        selected_machine.current_order = None

                        if self in selected_machine.queue:
                            selected_machine.queue.remove(self)

                        # Calculate break time and go to break
                        break_time = self.calculate_break_time(interuption.cause)
                        if break_time:
                            self.on_break = True
                            yield self.env.timeout(break_time)
                            self.on_break = False
                        continue

                    # Request worker for machine
                    try:
                        yield self.plant.worker_shift.get(selected_machine.number_of_worker)
                        self.working = True
                        self.machine = selected_machine
                        selected_machine.current_order = self
                        # Added for the case of order assignment by a rl agent
                        if self.plant.worker_shift.level > 3 and self.plant.env.now % (
                                5 * 3600) and self.plant.uninitialized_orders:
                            self.plant.step_event.succeed()
                            self.plant.step_event = self.plant.env.event()
                            self.plant.calculate_state()
                    except simpy.Interrupt as interuption:
                        selected_machine.current_order = None
                        # Calculate break time and go to break
                        break_time = self.calculate_break_time(interuption.cause)
                        if break_time:
                            self.on_break = True
                            yield self.env.timeout(break_time)
                            self.on_break = False
                        continue

                    # Setup machine for order
                    if not self.setup_complete:
                        try:
                            if not self.setup_time:
                                self.setup_time = self.get_setup_time(selected_machine)
                            yield self.env.timeout(self.setup_time)
                            self.setup_complete = True
                            self.setup_time = 0
                        except simpy.Interrupt as interuption:
                            if self in selected_machine.queue:
                                selected_machine.queue.remove(self)
                            selected_machine.current_order = None

                            # Calculate break time and go to break
                            break_time = self.calculate_break_time(interuption.cause)
                            if break_time:
                                self.on_break = True
                                yield self.env.timeout(break_time)
                                self.on_break = False
                            continue

                    # Start producing until finished or interruption
                    production_session_start = self.env.now
                    self.output_count_production_session = 0
                    try:
                        while self.output_count_order < self.number_of_pallets:
                            try:
                                unit_session_start = self.env.now
                                unit_processing_time = self.get_unit_processing_time()
                                yield self.env.timeout(unit_processing_time)
                                unit_session_end = self.env.now
                                self.output_count_order += 1
                                self.output_count_production_session += 1
                                self.machine.time_produced_today += unit_processing_time
                                self.plant.unit_sessions.append(dict(Task=selected_machine.machine_id,
                                                                     Start=transform_to_real_time(unit_session_start,
                                                                                                  self.plant.simulation_start),
                                                                     Finish=transform_to_real_time(unit_session_end,
                                                                                                   self.plant.simulation_start),
                                                                     Resource=self.name,
                                                                     Priority=self.priority))
                                self.plant.unit_sessions_today.append(dict(Task=selected_machine.machine_id,
                                                                           Start=transform_to_real_time(
                                                                               unit_session_start,
                                                                               self.plant.simulation_start),
                                                                           Finish=transform_to_real_time(
                                                                               unit_session_end,
                                                                               self.plant.simulation_start),
                                                                           Resource=self.name,
                                                                           Priority=self.priority,
                                                                           Revenue=self.unit_cost * self.boxes_per_pallet))
                                self.plant.revenue_step += self.unit_cost * self.boxes_per_pallet
                            except simpy.Interrupt as interuption:
                                raise interuption
                    except simpy.Interrupt as interuption:
                        if self in selected_machine.queue:
                            selected_machine.queue.remove(self)
                        selected_machine.current_order = None

                        production_session_end = self.env.now
                        self.plant.production_sessions.append(dict(Task=selected_machine.machine_id,
                                                                   Start=transform_to_real_time(
                                                                       production_session_start,
                                                                       self.plant.simulation_start),
                                                                   Finish=transform_to_real_time(production_session_end,
                                                                                                 self.plant.simulation_start),
                                                                   Resource=self.name,
                                                                   Priority=self.priority,
                                                                   Due_Date=self.due_date,
                                                                   Type=self.product_type.name))
                        self.working = False
                        # Calculate break time and go to break
                        break_time = self.calculate_break_time(interuption.cause)
                        if break_time:
                            self.on_break = True
                            yield self.env.timeout(break_time)
                            self.on_break = False
                        continue
                self.plant.worker_shift.put(selected_machine.number_of_worker)
                if self in selected_machine.queue:
                    selected_machine.queue.remove(self)
                selected_machine.current_order = None
                production_session_end = self.env.now
                self.plant.production_sessions.append(dict(Task=selected_machine.machine_id,
                                                           Start=transform_to_real_time(production_session_start,
                                                                                        self.plant.simulation_start),
                                                           Finish=transform_to_real_time(production_session_end,
                                                                                         self.plant.simulation_start),
                                                           Resource=self.name,
                                                           Priority=self.priority,
                                                           Due_Date=self.due_date,
                                                           Type=self.product_type.name))
                self.plant.open_orders.remove(self)
                self.plant.finished_orders.append(self)
                self.finished = True
                self.working = False
                # Event for the case of manuel scheduling but only if there are any new scheduling decisions to take
                if self.plant.uninitialized_orders:
                    self.plant.step_event.succeed()
                    self.plant.step_event = self.plant.env.event()
