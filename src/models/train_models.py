""" Define functions model training """
import time
from datetime import datetime

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.experimental import enable_halving_search_cv
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, mean_squared_error, r2_score
from tensorforce import Agent
from tqdm import tqdm

from src.models import run_agent, run_agent_mushroom
from src.utils import get_random_hyperparameter


def create_pipeline_model(classifier, numeric_features, categorical_features, feature_selector=None):
    """
    Create a pipeline to train a model
    :param classifier: Model to train
    :param numeric_features: List of numeric features
    :param categorical_features: List of categorical features
    :param feature_selector: Optional feature selector pipeline step
    :returns: Sklearn pipeline including data preprocessing steps, an optional feature selector and the model
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features)
            , ('categorical', categorical_transformer, categorical_features)
        ])
    if feature_selector:
        pipeline = Pipeline(steps=[["preprocessor", preprocessor],
                                   ["feature_selection", feature_selector],
                                   ['classifier', classifier]])
    else:
        pipeline = Pipeline(steps=[["preprocessor", preprocessor],
                                   ['classifier', classifier]])
    return pipeline


def train_pipeline_model(model, X, y, numeric_features, categorical_features, feature_selector=None):
    """
        Create a pipeline to train a model
        :param X: Feature
        :param y: Target variable
        :param model: Model to train
        :param numeric_features: List of numeric features
        :param categorical_features: List of categorical features
        :param feature_selector: Optional feature selector pipeline step
        :returns: Sklearn pipeline including data preprocessing steps, an optional feature selector and the model
        """
    start = time.time()
    time_clock = datetime.fromtimestamp(time.time())
    print(f"Start at {time_clock.hour}:{time_clock.minute}:{time_clock.second}")
    pipeline = create_pipeline_model(model, numeric_features, categorical_features, feature_selector)
    pipeline_fit = pipeline.fit(X, y.values.ravel())
    print(f"Duration: {time.time() - start}")
    return pipeline_fit


def evaluate_pipeline(pipeline, X, y, classification=True):
    """
    Evaluate given pipeline
    :param pipeline: Pipeline to be evaluated
    :param X: Feature values
    :param y: Target values
    :param classification: Flag if classification or regression metrics should be used for evaluation
    """
    start = time.time()
    time_clock = datetime.fromtimestamp(time.time())
    print(f"Start at {time_clock.hour}:{time_clock.minute}:{time_clock.second}")
    y_pred = pipeline.predict(X)
    if classification:
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy}")
        f1 = f1_score(y, y_pred, average="macro")
        print(f"F1: {f1}")
        precision = precision_score(y, y_pred, average="macro")
        print(f"Precision: {precision}")
        recall = recall_score(y, y_pred, average="macro")
        print(f"Recall: {recall}")
    else:
        rmse = mean_squared_error(y, y_pred, squared=False)
        print(f"RMSE: {round(rmse, 4)}")
        r2 = r2_score(y, y_pred)
        print(f"R^2: {round(r2, 4)}")

    print(f"Duration: {time.time() - start}")


@ignore_warnings(category=ConvergenceWarning)
def optimize_hyperparameters(X, y, model, feature_selector, numeric_features, categorical_features, hyperparameter,
                             random_state, scoring="accuracy"):
    """
        Create a pipeline to train a model
        :param X: Feature values
        :param y: Target values
        :param model: Model to train
        :param feature_selector: Optional feature selector pipeline step
        :param numeric_features: List of numeric features
        :param categorical_features: List of categorical features
        :param hyperparameter: Hyperparameter grid for optimization
        :param random_state: Random state for reproducibility
        :param scoring: Scoring metric for evaluation
        :returns: Hyperparameter optimization information
        """
    start = time.time()
    time_clock = datetime.fromtimestamp(time.time())
    print(f"Start at {time_clock.hour}:{time_clock.minute}:{time_clock.second}")
    pipeline = create_pipeline_model(model, numeric_features, categorical_features, feature_selector)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = HalvingGridSearchCV(estimator=pipeline, param_grid=hyperparameter, cv=inner_cv,
                              scoring=scoring, n_jobs=-1, random_state=random_state, return_train_score=True)
    clf_fit = clf.fit(X, y.values.ravel())
    print(f"Duration: {time.time() - start}")
    return clf_fit


def hyperparameter_tuning(environment, param_grid, k, due_date_range_list, number_orders_start_list,
                          average_count_new_orders_list, worker_list, random_states, random_seed):
    """
    Random  hyperparameter tuning for reinforcement learning models
    :param environment: Simulation environment to train or evaluate the agent
    :param param_grid: Hyperparameter grid
    :param k: Number of hyperparameter combination to test
    :param due_date_range_list: List of due date ranges for simulation
    :param number_orders_start_list: List of numbers orders start for simulation
    :param average_count_new_orders_list: List of average counts new orders for simulation
    :param worker_list: List of worker number for simulation
    :param random_states: List of random states for simulation
    :param random_seed: Random state for reproducibility
    :returns: Dataframe with results for hyperparameter grid
    """
    param_selected = get_random_hyperparameter(param_grid, k, random_seed)
    hyperparameter_results = []
    for param in tqdm(param_selected):
        # Define agent
        agent = Agent.create(
            agent='dqn',
            environment=environment,
            memory=90,
            batch_size=param["batch_size"],
            update_frequency=param["update_frequency"],
            summarizer=dict(
                directory='summaries', filename="summary_select",
                summaries=["action-value", "entropy", "graph", "kl-divergence", "loss", "parameters", "reward",
                           "update-norm", "updates", "variables"]
            ),
            horizon=param["horizon"],
            discount=param["discount"],
            return_processing=param["return_processing"],
            state_preprocessing=param["state_preprocessing"],
            reward_processing=param["reward_processing"],
            target_update_weight=param["target_update_weight"],
            exploration=0.1,
            l2_regularization=param["l2_regularization"]
        )

        # Train agent
        reward_training = run_agent(agent, environment, due_date_range_list, number_orders_start_list,
                                    average_count_new_orders_list,
                                    worker_list, random_states, episodes=1, evaluate=False)

        # Evaluate agent
        reward_evaluation = run_agent(agent, environment, [(3, 7)], number_orders_start_list,
                                      average_count_new_orders_list, worker_list, [1], episodes=1, evaluate=False)

        hyperparameter_results.append({"batch_size": param["batch_size"],
                                       "update_frequency": param["update_frequency"],
                                       "horizon": param["horizon"],
                                       "discount": param["discount"],
                                       "return_processing": param["return_processing"],
                                       "state_preprocessing": param["state_preprocessing"],
                                       "reward_processing": param["reward_processing"],
                                       "target_update_weight": param["target_update_weight"],
                                       "l2_regularization": param["l2_regularization"],
                                       "reward_training_mean": reward_training["reward"].mean(),
                                       "reward_training_var": reward_training["reward"].var(),
                                       "reward_evaluation_mean": reward_evaluation["reward"].mean(),
                                       "reward_evaluation_var": reward_evaluation["reward"].var(),
                                       "reward_df": reward_evaluation})
    return pd.DataFrame(hyperparameter_results)


def train_som(n_neurons, m_neurons, data, sigma, learning_rate, random_seed, number_iterations=10000):
    """
    Create and train a self organizing map
    :param n_neurons: Number of n neurons for self organizing maps
    :param m_neurons: Number of m neurons for self organizing maps
    :param data: Data for training the self organizing map
    :param sigma: Parameter sigma for training
    :param learning_rate: Parameter learning rate for training
    :param random_seed: Random state for reproducibility
    :param number_iterations: Number of iterations in training process
    :returns: Trained self organizing map and normalization function
    """
    columns = ['number_available_machines', 'number_of_jobs',
               'number_available_machines_1', 'number_of_jobs_1',
               'remaining_processing_time_1', 'number_available_machines_3',
               'number_of_jobs_3', 'remaining_processing_time_3',
               'number_available_machines_4', 'number_of_jobs_4',
               'remaining_processing_time_4', 'number_available_machines_5',
               'number_of_jobs_5', 'remaining_processing_time_5',
               'processing_time_max', 'remaining_processing_time_max',
               'slack_time_max', 'due_date_tightness_max', 'sojourn_time_max',
               'completion_rate_max', 'processing_time_min',
               'remaining_processing_time_min', 'slack_time_min',
               'due_date_tightness_min', 'sojourn_time_min', 'completion_rate_min',
               'processing_time_mean', 'remaining_processing_time_mean',
               'slack_time_mean', 'due_date_tightness_mean', 'sojourn_time_mean',
               'completion_rate_mean', 'processing_time_median',
               'remaining_processing_time_median', 'slack_time_median',
               'due_date_tightness_median', 'sojourn_time_median',
               'completion_rate_median', 'processing_time_variance',
               'remaining_processing_time_variance', 'slack_time_variance',
               'due_date_tightness_variance', 'sojourn_time_variance',
               'completion_rate_variance', 'processing_time_sum',
               'remaining_processing_time_sum', 'slack_time_sum',
               'due_date_tightness_sum', 'sojourn_time_sum', 'completion_rate_sum',
               'revenue_today', 'penalty_today', 'assign_priority_cr',
               'assign_priority_ds', 'assign_priority_edd', 'assign_priority_fifo',
               'assign_priority_lpt', 'assign_priority_mdd', 'assign_priority_spt',
               'assign_priority_srpt']
    X = data[columns]
    normalize_mean = np.mean(X, axis=0)
    normalize_std = np.std(X, axis=0)

    def normalize_data_som(data, normalize_mean=normalize_mean, normalize_std=normalize_std):
        """
        Normalize dataframe
        :param data: Data to normalize
        :param normalize_mean: Mean of columns to normalize
        :param normalize_std: Standard deviation of columns to normalize
        :returns: Normalized dataframe
        """
        return (data - normalize_mean) / normalize_std

    X = normalize_data_som(X)
    X = X.values

    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, X.shape[1], sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', random_seed=random_seed)

    som.pca_weights_init(X)
    som.train(X, number_iterations, verbose=True)  # random training

    return som, normalize_data_som


def hyperparameter_tuning_assign_ppo(environment, param_grid, k, due_date_range_list, number_orders_start_list,
                                     average_count_new_orders_list, worker_list, random_states, random_seed):
    """
    Random grid hyperparameter tuning for proximal policy optimization
    :param environment: Simulation environment
    :param param_grid: Hyperparameter grid to evaluate
    :param k: Number of hyperparameter grids to evaluate
    :param due_date_range_list: List of due date ranges for simulation
    :param number_orders_start_list: List of numbers orders start for simulation
    :param average_count_new_orders_list: List of average counts new orders for simulation
    :param worker_list: List of worker number for simulation
    :param random_states:List of random states for simulation
    :param random_seed: Random state for reproducibility
    :returns: Dataframe with hyperparameter tuning results
    """
    param_selected = get_random_hyperparameter(param_grid, k, random_seed)
    hyperparameter_results = []
    for param in tqdm(param_selected):
        # Define agent
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            memory=101000,
            batch_size=param["batch_size"],
            update_frequency=param["update_frequency"],
            summarizer=dict(
                directory='summaries', filename="summary_select",
                summaries=["action-value", "entropy", "graph", "kl-divergence", "loss", "parameters", "reward",
                           "update-norm", "updates", "variables"]
            ),
            discount=param["discount"],
            return_processing=param["return_processing"],
            state_preprocessing=param["state_preprocessing"],
            reward_processing=param["reward_processing"],
            exploration=0.1,
            max_episode_timesteps=1000,
            l2_regularization=param["l2_regularization"],
            likelihood_ratio_clipping=param["likelihood_ratio_clipping"],
            entropy_regularization=param["entropy_regularization"]
        )

        # Train agent
        reward_training = run_agent(agent, environment, due_date_range_list, number_orders_start_list,
                                    average_count_new_orders_list,
                                    worker_list, random_states, episodes=1, evaluate=False)

        # Evaluate agent
        reward_evaluation = run_agent(agent, environment, [(3, 7)], number_orders_start_list,
                                      average_count_new_orders_list, worker_list, [1], episodes=1, evaluate=False)

        hyperparameter_results.append({"batch_size": param["batch_size"],
                                       "update_frequency": param["update_frequency"],
                                       "discount": param["discount"],
                                       "return_processing": param["return_processing"],
                                       "state_preprocessing": param["state_preprocessing"],
                                       "reward_processing": param["reward_processing"],
                                       "l2_regularization": param["l2_regularization"],
                                       "entropy_regularization": param["entropy_regularization"],
                                       "likelihood_ratio_clipping": param["likelihood_ratio_clipping"],
                                       "reward_training_mean": reward_training["reward"].mean(),
                                       "reward_training_var": reward_training["reward"].var(),
                                       "reward_evaluation_mean": reward_evaluation["reward"].mean(),
                                       "reward_evaluation_var": reward_evaluation["reward"].var(),
                                       "reward_df": reward_evaluation})
    return pd.DataFrame(hyperparameter_results)


def hyperparameter_tuning_assign_ac(environment, param_grid, k, due_date_range_list, number_orders_start_list,
                                    average_count_new_orders_list, worker_list, random_states, random_seed):
    """
    Random grid hyperparameter tuning for actor critic reinforcement learning
    :param environment: Simulation environment
    :param param_grid: Hyperparameter grid to evaluate
    :param k: Number of hyperparameter grids to evaluate
    :param due_date_range_list: List of due date ranges for simulation
    :param number_orders_start_list: List of numbers orders start for simulation
    :param average_count_new_orders_list: List of average counts new orders for simulation
    :param worker_list: List of worker number for simulation
    :param random_states:List of random states for simulation
    :param random_seed: Random state for reproducibility
    :returns: Dataframe with hyperparameter tuning results
    """
    param_selected = get_random_hyperparameter(param_grid, k, random_seed)
    hyperparameter_results = []
    for param in tqdm(param_selected):
        # Define agent
        agent = Agent.create(
            agent='ac',
            environment=environment,
            memory=11000,
            batch_size=param["batch_size"],
            update_frequency=param["update_frequency"],
            max_episode_timesteps=1000,
            critic_optimizer=param["critic_optimizer"],
            summarizer=dict(
                directory='summaries', filename="summary_select",
                summaries=["action-value", "entropy", "graph", "kl-divergence", "loss", "parameters", "reward",
                           "update-norm", "updates", "variables"]
            ),
            horizon=param["horizon"],
            discount=param["discount"],
            return_processing=param["return_processing"],
            state_preprocessing=param["state_preprocessing"],
            reward_processing=param["reward_processing"],
            exploration=0.1,
            l2_regularization=param["l2_regularization"],
            entropy_regularization=param["entropy_regularization"]
        )

        # Train agent
        reward_training = run_agent(agent, environment, due_date_range_list, number_orders_start_list,
                                    average_count_new_orders_list,
                                    worker_list, random_states, episodes=1, evaluate=False)

        # Evaluate agent
        reward_evaluation = run_agent(agent, environment, [(3, 7)], number_orders_start_list,
                                      average_count_new_orders_list, worker_list, [1], episodes=1, evaluate=False)

        hyperparameter_results.append({"batch_size": param["batch_size"],
                                       "update_frequency": param["update_frequency"],
                                       "critic_optimizer": param["critic_optimizer"],
                                       "discount": param["discount"],
                                       "horizon": param["horizon"],
                                       "discount": param["discount"],
                                       "return_processing": param["return_processing"],
                                       "state_preprocessing": param["state_preprocessing"],
                                       "reward_processing": param["reward_processing"],
                                       "l2_regularization": param["l2_regularization"],
                                       "entropy_regularization": param["entropy_regularization"],
                                       "reward_training_mean": reward_training["reward"].mean(),
                                       "reward_training_var": reward_training["reward"].var(),
                                       "reward_evaluation_mean": reward_evaluation["reward"].mean(),
                                       "reward_evaluation_var": reward_evaluation["reward"].var(),
                                       "reward_df": reward_evaluation})
    return pd.DataFrame(hyperparameter_results)


def hyperparameter_tuning_mushroom(param_grid, k, data, product_types_df, machines_df, orders_df, simulation_start,
                                   priority_rules, due_date_range_list, number_orders_start_list,
                                   average_count_new_orders_list, worker_list, random_states, random_seed):
    """
    Random hyperparameter grid search
    :param param_grid: Hyperparameter grid to evaluate
    :param k: Number of hyperparameter grids to evaluate
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
    :param random_seed: Random state for reproducibility
    :returns: Dataframe with hyperparameter tuning results
    """
    param_selected = get_random_hyperparameter(param_grid, k, random_seed)
    hyperparameter_results = []
    for params in tqdm(param_selected):
        # Train agent
        reward_list = run_agent_mushroom(data, product_types_df, machines_df, orders_df, simulation_start,
                                         priority_rules,
                                         due_date_range_list, number_orders_start_list, average_count_new_orders_list,
                                         worker_list,
                                         random_states, params, episodes=1)
        reward_evaluation = pd.DataFrame(reward_list,
                                         columns=["state_previous", "action", "reward", "state_after", "NA", "NA_2"])

        hyperparameter_results.append({"epsilon_param": params["epsilon_param"],
                                       "learning_rate": params["learning_rate"],
                                       "number_of_states": params["number_of_states"],
                                       "sigma": params["sigma"],
                                       "learning_rate_som": params["learning_rate_som"],
                                       "reward_evaluation_mean": reward_evaluation["reward"].mean(),
                                       "reward_evaluation_var": reward_evaluation["reward"].var(),
                                       "reward_df": reward_evaluation})
    return pd.DataFrame(hyperparameter_results)


def hyperparameter_tuning_assign_deepq(environment, param_grid, k, due_date_range_list, number_orders_start_list,
                                       average_count_new_orders_list, worker_list, random_states, random_seed):
    """
    Random grid hyperparameter tuning for deep q learning
    :param environment: Simulation environment
    :param param_grid: Hyperparameter grid to evaluate
    :param k: Number of hyperparameter grids to evaluate
    :param due_date_range_list: List of due date ranges for simulation
    :param number_orders_start_list: List of numbers orders start for simulation
    :param average_count_new_orders_list: List of average counts new orders for simulation
    :param worker_list: List of worker number for simulation
    :param random_states:List of random states for simulation
    :param random_seed: Random state for reproducibility
    :returns: Dataframe with hyperparameter tuning results
    """
    param_selected = get_random_hyperparameter(param_grid, k, random_seed)
    hyperparamter_results = []
    for param in tqdm(param_selected):
        # Define agent
        agent = Agent.create(
            agent='dqn',
            environment=environment,
            memory=240,
            batch_size=param["batch_size"],
            update_frequency=param["update_frequency"],
            summarizer=dict(
                directory='summaries', filename="summary_select",
                summaries=["action-value", "entropy", "graph", "kl-divergence", "loss", "parameters", "reward",
                           "update-norm", "updates", "variables"]
            ),
            horizon=param["horizon"],
            discount=param["discount"],
            return_processing=param["return_processing"],
            state_preprocessing=param["state_preprocessing"],
            reward_processing=param["reward_processing"],
            target_update_weight=param["target_update_weight"],
            exploration=0.1,
            l2_regularization=param["l2_regularization"]
        )

        # Train agent
        reward_training = run_agent(agent, environment, due_date_range_list, number_orders_start_list,
                                    average_count_new_orders_list,
                                    worker_list, random_states, episodes=1, evaluate=False)

        # Evaluate agent
        reward_evaluation = run_agent(agent, environment, [(3, 7)], number_orders_start_list,
                                      average_count_new_orders_list, worker_list, [1], episodes=1, evaluate=False)

        hyperparamter_results.append({"batch_size": param["batch_size"],
                                      "update_frequency": param["update_frequency"],
                                      "horizon": param["horizon"],
                                      "discount": param["discount"],
                                      "return_processing": param["return_processing"],
                                      "state_preprocessing": param["state_preprocessing"],
                                      "reward_processing": param["reward_processing"],
                                      "target_update_weight": param["target_update_weight"],
                                      "l2_regularization": param["l2_regularization"],
                                      "reward_training_mean": reward_training["reward"].mean(),
                                      "reward_training_var": reward_training["reward"].var(),
                                      "reward_evaluation_mean": reward_evaluation["reward"].mean(),
                                      "reward_evaluation_var": reward_evaluation["reward"].var(),
                                      "reward_df": reward_evaluation})
    return pd.DataFrame(hyperparamter_results)
