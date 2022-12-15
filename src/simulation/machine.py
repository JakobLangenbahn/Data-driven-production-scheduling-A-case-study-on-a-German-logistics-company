""" Define machine object for simulation """
from random import seed, expovariate
import simpy
from src.utils import generate_random_deviation


class ProductType(object):
    """
    Class for saving product type information
    :param product_type_id: Unique identifier of the product type
    :param name: Name of the product type
    :param setup_time: Duration required for setup in hours
    :param number_of_worker: Number of worker required to work on a product of this type
    """

    def __init__(self, product_type_id, name, setup_time, number_of_worker):
        """
        Constructor method
        """
        self.product_type_id = product_type_id
        self.name = name
        self.setup_time = setup_time
        self.number_of_worker = number_of_worker


class Machine(object):
    """
    Class for saving product type information
    :param env: Simpy environment in which the machine is used
    :param machine_id: Unique identifier of the machine
    :param product_type: Product type class of the machine
    :param mmtf: Mean time to failure of machine
    :param random_state: Random seed for reproducibility
    :param verbose: Flag if information should be printed
    """

    def __init__(self, env, machine_id, product_type, mmtf=60 * 60 * 24 * 7, random_state=42, verbose=False):
        """
        Constructor method
        """
        # Initialize environment
        self.env = env
        self.resource = simpy.PriorityResource(env, capacity=1)
        self.random_state = random_state
        # Initialize static attributes
        self.machine_id = machine_id
        self.product_type = product_type
        self.setup_time = self.product_type.setup_time
        self.mttf = mmtf
        self.number_of_worker = self.product_type.number_of_worker
        # Initialize dynamic attributes
        self.available = True
        self.current_order = None
        self.time_produced_today = 0
        # Initialize function attributes
        self.verbose = verbose
        # Initialize data collection container
        self.queue = []
        self.last_order_getting_machine = []
        self.last_order_producing_machine = []
        # Initialize breakdown process
        env.process(self.breakdown())

    def breakdown(self):
        """
        Stochastic process for breakdowns of machines
        """
        break_mean = 1 / self.mttf
        i = 1
        while True:
            # Set seed in a way that it is repeatable but also different for each machine
            seed(self.random_state + i * self.machine_id)
            ttf = expovariate(break_mean)
            repair_time = generate_random_deviation(5 * 60, 180 * 60, 70 * 60, 35 * 60,
                                                    int(self.random_state + i * self.machine_id))
            i += 1
            yield self.env.timeout(ttf)
            if self.verbose:
                print(f"Machine {self.machine_id} breakdown at {self.env.now}")
            self.available = False
            if self.current_order and not self.current_order.on_break:
                if self.verbose:
                    print(f"Interrupt {self.current_order.name} at {self.env.now}")
                self.current_order.process.interrupt(cause="Machine breakdown")
            yield self.env.timeout(repair_time)
            self.available = True
