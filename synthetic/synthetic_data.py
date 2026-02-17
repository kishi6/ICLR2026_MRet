import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from utils import fix_seed
from algo import RetentionPredictionModel_XGB, user_retain_func
import conf


def generate_reward_data(
    dim: int = 10,
    T: float = 1.0,
    alpha_param: tuple = None,
    beta_param: tuple = None,
    random_state: int = 0,
):
    fix_seed(random_state)
    random_ = check_random_state(random_state)
    x = random_.standard_normal((5000, dim))
    y = random_.standard_normal((5000, dim))
    alpha_matrix = random_.standard_normal((dim, 1))
    beta_matrix = random_.standard_normal((dim, 1))
    alpha_x = x @ alpha_matrix
    beta_x = x @ beta_matrix
    alpha_mmscaler = StandardScaler()
    alpha_mmscaler.fit(alpha_x)
    beta_mmscaler = StandardScaler()
    beta_mmscaler.fit(beta_x)
    reward_data = {
        "alpha_matrix": alpha_matrix,
        "beta_matrix": beta_matrix,
        "alpha_mmscaler": alpha_mmscaler,
        "beta_mmscaler": beta_mmscaler,
    }
    return reward_data


def train_model(
    dim: int = 10,
    T: float = 1.0,
    n_train: int = 1000,
    reward_data: dict = None,
    alpha_param: tuple = None,
    beta_param: tuple = None,
    random_state: int = 0,
):
    fix_seed(random_state)
    random_ = check_random_state(random_state)
    x = random_.standard_normal((n_train, dim))
    y = random_.standard_normal((n_train, dim))
    alpha_x = x @ reward_data["alpha_matrix"]
    alpha_y = y @ reward_data["alpha_matrix"]
    beta_x = x @ reward_data["beta_matrix"]
    beta_y = y @ reward_data["beta_matrix"]
    alpha_x = reward_data["alpha_mmscaler"].transform(alpha_x)
    alpha_y = reward_data["alpha_mmscaler"].transform(alpha_y)
    beta_x = reward_data["beta_mmscaler"].transform(beta_x)
    beta_y = reward_data["beta_mmscaler"].transform(beta_y)
    alpha_x = alpha_param[1] * alpha_x + alpha_param[0]
    alpha_y = alpha_param[1] * alpha_y + alpha_param[0]
    beta_x = beta_param[1] * beta_x + beta_param[0]
    beta_y = beta_param[1] * beta_y + beta_param[0]
    if conf.reward_shape == "linear":
        alpha_x = 1 / alpha_x
        alpha_y = 1 / alpha_y
    elif conf.reward_shape in ["semi_quadratic", "quadratic"]:
        beta_x = (beta_x - 1) / (alpha_x ** 2)
        beta_y = (beta_y - 1) / (alpha_y ** 2)
    m_x = 2 * alpha_param[1] * random_.standard_normal((n_train, 1)) + alpha_param[0]
    m_y = 2 * alpha_param[1] * random_.standard_normal((n_train, 1)) + alpha_param[0]
    u_x = user_retain_func(alpha_x, beta_x, m_x)
    u_y = user_retain_func(alpha_y, beta_y, m_y)
    flag_x = random_.rand(n_train, 1) < u_x
    flag_y = random_.rand(n_train, 1) < u_y
    input_x = np.concatenate([x, m_x], axis=1)
    input_y = np.concatenate([y, m_y], axis=1)
    model = RetentionPredictionModel_XGB(num_features=6, random_state=random_state, random_=random_)
    X_combined = np.vstack([input_x, input_y])
    y_combined = np.vstack([flag_x, flag_y]).ravel().astype(int)
    model.fit(X_combined, y_combined)
    return model


def generate_data(
    n_x: int = 1000,
    n_y: int = 1000,
    dim: int = 10,
    rel_noise: float = 0.0,
    T: float = 1.0,
    K: int = 5,
    kappa: float = 0.5,
    reward_data: dict = None,
    alpha_param: tuple = None,
    beta_param: tuple = None,
    random_state: int = 0,
    random_: np.random.RandomState = None,
):
    x = random_.standard_normal((n_x, dim))
    y = random_.standard_normal((n_y, dim))
    alpha_x = x @ reward_data["alpha_matrix"]
    alpha_y = y @ reward_data["alpha_matrix"]
    beta_x = x @ reward_data["beta_matrix"]
    beta_y = y @ reward_data["beta_matrix"]
    alpha_x = reward_data["alpha_mmscaler"].transform(alpha_x)
    alpha_y = reward_data["alpha_mmscaler"].transform(alpha_y)
    beta_x = reward_data["beta_mmscaler"].transform(beta_x)
    beta_y = reward_data["beta_mmscaler"].transform(beta_y)
    alpha_x = alpha_param[1] * alpha_x + alpha_param[0]
    alpha_y = alpha_param[1] * alpha_y + alpha_param[0]
    beta_x = beta_param[1] * beta_x + beta_param[0]
    beta_y = beta_param[1] * beta_y + beta_param[0]
    if conf.reward_shape == "linear":
        alpha_x = 1 / alpha_x
        alpha_y = 1 / alpha_y
    elif conf.reward_shape in ["semi_quadratic", "quadratic"]:
        beta_x = (beta_x - 1) / (alpha_x ** 2)
        beta_y = (beta_y - 1) / (alpha_y ** 2)
    norms_x = np.linalg.norm(x, axis=1).reshape(-1, 1)
    norms_y = np.linalg.norm(y, axis=1).reshape(1, -1)
    dot_products = np.dot(x, y.T)
    cosine_similarity_matrix = dot_products / (norms_x * norms_y)
    rel_mat_true = (cosine_similarity_matrix + 1) / 2
    pop_x = random_.uniform(0, 1, size=(n_x, 1))
    pop_y = random_.uniform(0, 1, size=(n_y, 1))
    pop_matrix = pop_x @ pop_y.T
    rel_mat_true = (1 - kappa) * rel_mat_true + kappa * pop_matrix


    if conf.time_popularity:
        # Generate time-series popularity for users (pop_x) and items (pop_y)
        pop_x_time, user_type_x = generate_pop_time_series(random_, T, n_x, max_slope=0.5)
        pop_y_time, user_type_y = generate_pop_time_series(random_, T, n_y, max_slope=0.5)

        # Create pop_matrix of (time_step, n_x, n_y) using outer product
        # pop_x: (T, n_x), pop_y: (T, n_y)
        # => Outer product for each t using einsum('ti,tj->tij')
        pop_matrix_time = np.einsum('ti,tj->tij', pop_x_time, pop_y_time)  # (T, n_x, n_y)

        # Final rel_mat_true with time_step
        # Original rel_mat_true is referred to as rel_mat_true_base
        rel_mat_true_base = (cosine_similarity_matrix + 1) / 2
        rel_mat_true_time = (1 - kappa) * rel_mat_true_base[None, :, :] + kappa * pop_matrix_time
    else:
        rel_mat_true_time = None


    if rel_noise > 0.0:
        rel_mat_obs = np.copy(rel_mat_true)
        rel_mat_obs += random_.uniform(-rel_noise, rel_noise, size=(n_x, n_y))
        rel_mat_obs = np.maximum(rel_mat_obs, 0.001)
    else:
        rel_mat_obs = rel_mat_true
    dataset = {
        "T": T,
        "K": K,
        "x": x,
        "y": y,
        "n_x": n_x,
        "n_y": n_y,
        "alpha_x": alpha_x,
        "alpha_y": alpha_y,
        "beta_x": beta_x,
        "beta_y": beta_y,
        "rel_mat_true": rel_mat_true,
        "rel_mat_obs": rel_mat_obs,
        "rel_mat_true_time": rel_mat_true_time,
    }
    return dataset



def generate_pop_time_series(random_, time_step, n, max_slope=0.5):
    """
    Function to generate time-series popularity by dividing into 3 user types (increasing, decreasing, constant)

    Returns:
        pop (time_step, n): Time-series popularity for each user (clipped to 0-1)
        user_type (n,): 0=constant, 1=increasing, 2=decreasing
    """
    # Initial value (popularity around t=0)
    base = random_.uniform(0, 1, size=n)          # (n,)

    # Which user is which type (0: constant, 1: increasing, 2: decreasing)
    user_type = random_.choice([0, 1, 2], size=n) # (n,)

    # Slope (how much it increases/decreases)
    slope = random_.uniform(0, max_slope, size=n) # (n,)

    # Normalized time t to [0, 1] (T, 1)
    t = np.linspace(0.0, 1.0, time_step)[:, None]  # (T, 1)

    # Array to store results
    pop = np.zeros((time_step, n), dtype=float)    # (T, n)

    # Calculate for each type
    for i in range(n):
        if user_type[i] == 0:
            # constant: stays the same
            pop[:, i] = base[i]
        elif user_type[i] == 1:
            # increasing: increases with time
            pop[:, i] = base[i] + slope[i] * t[:, 0]
        else:
            # decreasing: decreases with time
            pop[:, i] = base[i] - slope[i] * t[:, 0]

    # Clip to range 0-1
    pop = np.clip(pop, 0.0, 1.0)

    return pop, user_type
