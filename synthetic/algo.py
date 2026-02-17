import numpy as np
from scipy.stats import rankdata as rank
import itertools
import conf
from xgboost import XGBClassifier


def naive_ranking(rel_vec_obs_t: np.ndarray, e: np.ndarray, active_users: np.ndarray) -> np.ndarray:
    ranked_indices = rank(-rel_vec_obs_t[active_users], method="ordinal") - 1
    ranked_exposures = np.zeros_like(rel_vec_obs_t)
    ranked_exposures[active_users] = e[ranked_indices]
    return ranked_exposures


def uniform_ranking(rel_vec_obs_t: np.ndarray, e: np.ndarray, active_users: np.ndarray, random_) -> np.ndarray:
    random_list = random_.uniform(0, 1, len(e))
    ranked_indices = rank(-random_list[active_users], method="ordinal") - 1
    ranked_exposures = np.zeros_like(rel_vec_obs_t)
    ranked_exposures[active_users] = e[ranked_indices]
    return ranked_exposures


def calc_fair(match: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    adjusted_expo = exposure / np.where(match == 0, 0.001, match)
    err_t = np.tile(adjusted_expo, (exposure.shape[0], 1))
    err_t -= adjusted_expo[np.newaxis].T
    err_t = np.abs(err_t.min(0))
    return err_t



def fairco_ranking(
    active_users: np.ndarray,
    merit: np.ndarray,
    rel_vec_obs_t: np.ndarray,
    exposure: np.ndarray,
    tau: int,
    e: np.ndarray,
    lam: float = 0.01,
) -> np.ndarray:
    err_t = np.zeros_like(merit)
    if tau > 0:
        adjusted_expo = exposure / merit
        adjusted_expo = np.nan_to_num(adjusted_expo, nan=0.0, posinf=0.0, neginf=0.0)
        err_t = adjusted_expo.max() - adjusted_expo
    adjusted_rel_t = rel_vec_obs_t + lam * tau * err_t
    ranked_indices = rank(-adjusted_rel_t[active_users], method="ordinal") - 1
    ranked_exposures = np.zeros_like(adjusted_rel_t)
    ranked_exposures[active_users] = e[ranked_indices]
    return ranked_exposures


def expfair_ranking(
    active_users: np.ndarray,
    merit: np.ndarray,
    rel_vec_obs_t: np.ndarray,
    exposure: np.ndarray,
    tau: int,
    e: np.ndarray,
    lam: float = 0.01,
) -> np.ndarray:
    err_t = np.zeros_like(merit)
    if tau > 0:
        adjusted_expo = exposure
        adjusted_expo = np.nan_to_num(adjusted_expo, nan=0.0, posinf=0.0, neginf=0.0)
        err_t = adjusted_expo.max() - adjusted_expo
    adjusted_rel_t = rel_vec_obs_t + lam * tau * err_t
    ranked_indices = rank(-adjusted_rel_t[active_users], method="ordinal") - 1
    ranked_exposures = np.zeros_like(adjusted_rel_t)
    ranked_exposures[active_users] = e[ranked_indices]
    return ranked_exposures


def predict_match(
    user,
    item,
    m_user,
    m_item,
    r,
    r_hat,
    A,
    alpha_max,
    model,
    method,
):
    m_user_ = m_user + A * r
    user_tile = np.repeat(user[None, :], m_user_.shape[0], axis=0)
    x_features_ = np.concatenate([user_tile, m_user_.reshape(-1, 1)], axis=1)
    y_features_ = np.concatenate([item, (m_item + alpha_max * r).reshape(-1, 1)], axis=1)
    u_user_ = np.clip(model.predict(x_features_, method), 0, 1)
    u_item_ = np.clip(model.predict(y_features_, method), 0, 1)
    return u_user_, u_item_


def user_retain_ranking(
    e,
    u_user,
    u_item,
    u_user_,
    u_item_,
    active_users,
    A,
    alpha_max,
    alpha_min,
    eta_user,
    eta_item
):
    adjusted_rel_t = eta_user * (1 / A * u_user_  - 1/ alpha_min *u_user )+  eta_item * 1 / alpha_max * (u_item_ - u_item)
    # adjusted_rel_t = eta_user * 1 / A * (u_user_ - u_user)+  eta_item * 1 / alpha_max * (u_item_ - u_item)
    ranked_indices = rank(-adjusted_rel_t[active_users], method="ordinal") - 1
    ranked_exposures = np.zeros_like(adjusted_rel_t)
    ranked_exposures[active_users] = e[ranked_indices]
    return ranked_exposures



def user_retain_func(alpha, beta, m):
    if conf.reward_shape == "linear":
        u = np.where(alpha * m + beta <= 0.95, alpha * m + beta, 1 - 0.05 * np.exp(2 * (0.95 - beta) / alpha - m))
    elif conf.reward_shape == "quadratic":
        u = beta * (m - alpha) ** 2 + 1.0
    elif conf.reward_shape == "semi_quadratic":
        u = np.where(m <= alpha, beta * (m - alpha) ** 2 + 0.95, 1 - 0.05 * np.exp(2 * (alpha - m)))
    else:
        u = m
    return np.clip(u, 0, 1)


def optimal_ranking(
    e,
    alpha_user,
    beta_user,
    alpha_item,
    beta_item,
    m_user,
    m_item,
    rel_vec_true_t,
    active_users,
    K,
):
    candidates = np.array(active_users)
    n_candidates = len(candidates)
    best_reward = -np.inf
    best_order = None
    actual_K = min(K, n_candidates)
    m_item_full = m_item
    for perm in itertools.permutations(candidates, actual_K):
        exposure = np.zeros(n_candidates)
        for i, idx in enumerate(perm):
            exposure[np.where(candidates == idx)[0][0]] = e[i]
        m_item_cand = m_item_full[candidates]
        alpha_item_cand = alpha_item[candidates]
        beta_item_cand = beta_item[candidates]
        match_user = np.sum(exposure * rel_vec_true_t[candidates]) + m_user
        match_item = exposure * rel_vec_true_t[candidates] + m_item_cand
        reward_user = user_retain_func(alpha_user, beta_user, match_user)
        reward_item = user_retain_func(alpha_item_cand, beta_item_cand, match_item)
        total_reward = reward_user + np.sum(reward_item)
        if total_reward > best_reward:
            best_reward = total_reward
            best_order = perm
    ranked_exposures = np.zeros_like(m_item_full)
    if best_order is not None:
        for i, idx in enumerate(best_order):
            ranked_exposures[idx] = e[i]
    return ranked_exposures


class RetentionPredictionModel_XGB:
    def __init__(self, num_features, random_state=12345, random_=None):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=random_state,
        )
        self.num_features = num_features
        self.X_all = []
        self.y_all = []
        self.classes_ = np.array([0, 1])
        self.loss_history = []
        self.initialized = False

    def fit(self, X, y):
        if len(X) == 0:
            return
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)
        self.X_all.append(X)
        self.y_all.append(y)
        X_total = np.vstack(self.X_all)
        y_total = np.concatenate(self.y_all)
        if 0 not in y_total:
            X_total = np.vstack([X_total, np.zeros((1, self.num_features))])
            y_total = np.append(y_total, 0)
        self.model.fit(X_total, y_total, eval_set=[(X_total, y_total)], verbose=False)
        self.initialized = True
        evals_result = self.model.evals_result()
        self.loss_history = evals_result["validation_0"]["logloss"]

    def predict(self, X, method=None):
        if not self.initialized or len(X) == 0:
            return np.full(len(X), 0.5)
        X = np.asarray(X)
        return self.model.predict_proba(X)[:, 1]
