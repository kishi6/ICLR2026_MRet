import numpy as np
from tqdm import tqdm
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from utils import exam_func
from algo import (
    naive_ranking,
    uniform_ranking,
    fairco_ranking,
    user_retain_ranking,
    user_retain_func,
    predict_match,
    calc_fair,
    optimal_ranking,
    expfair_ranking,
)
import conf

def run_dynamic_match(
    dataset=None,
    model=None,
    proportion=0.5,
    reward_type=None,
    ranking_metric=None,
    noise=None,
    results=None,
    lambda_=0.1,
    candidate_retention=0.002,
    random_state=12345,
):
    n_x = dataset["n_x"]
    n_y = dataset["n_y"]
    T = dataset["T"]
    K = dataset["K"]
    rel_mat_true = dataset["rel_mat_true"]
    rel_mat_obs = dataset["rel_mat_obs"]

    x = dataset["x"]
    y = dataset["y"]
    alpha_x = dataset["alpha_x"]
    alpha_y = dataset["alpha_y"]
    beta_x = dataset["beta_x"]
    beta_y = dataset["beta_y"]

    buffer_input = []
    buffer_labels = []
    alpha_x_ = alpha_x.copy()
    beta_x_ = beta_x.copy()
    alpha_y_ = alpha_y.copy()
    beta_y_ = beta_y.copy()
    merit_x = rel_mat_true.mean(axis=0)
    merit_y = rel_mat_true.mean(axis=1)

    rel_mat_true_time = dataset["rel_mat_true_time"]

    for method in results.keys():
        random_ = check_random_state(random_state)
        results[method]["active_users_x"][0] = np.ones(n_x, dtype=bool)
        results[method]["active_users_y"][0] = np.ones(n_y, dtype=bool)
        results[method]["true_user_retain_x"][0] = user_retain_func(
            alpha_x.reshape(-1), beta_x.reshape(-1), results[method]["match_x"][0]
        )
        results[method]["true_user_retain_y"][0] = user_retain_func(
            alpha_y.reshape(-1), beta_y.reshape(-1), results[method]["match_y"][0]
        )

        for t in tqdm(range(1, T), desc="time step"):
            if conf.time_popularity:
                rel_mat_true = rel_mat_true_time[t-1]
                rel_mat_obs = rel_mat_true_time[t-1]

            active_x_indices = np.where(results[method]["active_users_x"][t - 1])[0]
            active_y_indices = np.where(results[method]["active_users_y"][t - 1])[0]

            if len(active_x_indices) == 0 or len(active_y_indices) == 0:
                results[method]["match_x"][t] = results[method]["match_x"][t - 1]
                results[method]["match_y"][t] = results[method]["match_y"][t - 1]
                continue

            if random_.rand() < proportion:
                e = exam_func(n_y, K, shape=ranking_metric)
                A = np.sum(e[:K])
                alpha_max = np.max(e[:K])
                alpha_min = np.min(e[:K])

                x_t = random_.choice(active_x_indices)
                rel_vec_true_t = rel_mat_true[x_t]
                rel_vec_obs_t = rel_mat_obs[x_t]
                merit = merit_y

                if reward_type == "n_match":
                    match_x = results[method]["match_x"][t - 1][x_t] + A * rel_vec_obs_t
                    match_y = results[method]["match_y"][t - 1] + alpha_max * rel_vec_obs_t
                elif reward_type == "n_match_per":
                    match_x = (results[method]["match_x"][t - 1][x_t] * (t - 1) + A * rel_vec_obs_t) / t
                    match_y = (results[method]["match_y"][t - 1] * (t - 1) + alpha_max * rel_vec_obs_t) / t

                if method == "Uniform":
                    expo_alloc = uniform_ranking(rel_vec_obs_t, e, active_y_indices, random_)
                elif method == "MaxMatch":
                    expo_alloc = naive_ranking(rel_vec_obs_t, e, active_y_indices)
                elif method == "FairCo":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=lambda_,
                    )
                elif method == "FairCo (lam=0.01)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.01,
                    )
                elif method == "FairCo (lam=0.1)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.1,
                    )
                elif method == "FairCo (lam=100)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=100,
                    )
                elif method == "FairCo (equal exposure) (lam=0.1)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.1,
                    )
                elif method == "FairCo (equal exposure) (lam=1)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.1,
                    )
                elif method == "FairCo (equal exposure) (lam=10)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=10,
                    )
                elif method == "FairCo (equal exposure) (lam=100)":
                    expo_alloc = fairco_ranking(
                        active_y_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_y"][t - 1],
                        tau=t,
                        e=e,
                        lam=100,
                    )
                elif method == "MRet":
                    u_x_, u_y_ = predict_match(
                        user=x[x_t],
                        item=y,
                        m_user=results[method]["match_x"][t - 1][x_t],
                        m_item=results[method]["match_y"][t - 1],
                        r_hat=rel_vec_obs_t,
                        r=rel_vec_obs_t,
                        A=A,
                        alpha_max=alpha_max,
                        model=model,
                        method=method,
                    )
                    expo_alloc = user_retain_ranking(
                        e=e,
                        u_user=results[method]["user_retain_x"][t - 1][x_t],
                        u_item=results[method]["user_retain_y"][t - 1],
                        u_user_=u_x_,
                        u_item_=u_y_,
                        active_users=active_y_indices,
                        A=A,
                        alpha_max=alpha_max,
                        alpha_min=alpha_min,
                        eta_user=conf.eta_male,
                        eta_item=conf.eta_female,
                        
                    )
                elif method == "optimal_ranking":
                    expo_alloc = optimal_ranking(
                        e=e,
                        alpha_user=alpha_x[x_t],
                        beta_user=beta_x[x_t],
                        alpha_item=alpha_y.reshape(-1),
                        beta_item=beta_y.reshape(-1),
                        m_user=results[method]["match_x"][t - 1][x_t],
                        m_item=results[method]["match_y"][t - 1],
                        rel_vec_true_t=rel_vec_true_t,
                        active_users=active_y_indices,
                        K=K,
                    )
                elif method == "MRet (noise)":
                    true_u_x_ = user_retain_func(alpha_x[x_t], beta_x[x_t], match_x)
                    true_u_y_ = user_retain_func(alpha_y.reshape(-1), beta_y.reshape(-1), match_y)
                    expo_alloc = user_retain_ranking(
                        e=e,
                        u_user=results[method]["true_user_retain_x"][t - 1][x_t],
                        u_item=results[method]["true_user_retain_y"][t - 1],
                        u_user_=random_.normal(true_u_x_, noise, size=true_u_x_.shape),
                        u_item_=random_.normal(true_u_y_, noise, size=true_u_y_.shape),
                        active_users=active_y_indices,
                        A=A,
                        alpha_max=alpha_max,
                        alpha_min=alpha_min,
                        eta_user=conf.eta_male,
                        eta_item=conf.eta_female,
                    )
                elif method == "MRet (best)":
                    true_u_x_ = user_retain_func(alpha_x[x_t], beta_x[x_t], match_x)
                    true_u_y_ = user_retain_func(alpha_y.reshape(-1), beta_y.reshape(-1), match_y)
                    expo_alloc = user_retain_ranking(
                        e=e,
                        u_user=results[method]["true_user_retain_x"][t - 1][x_t],
                        u_item=results[method]["true_user_retain_y"][t - 1],
                        u_user_=true_u_x_,
                        u_item_=true_u_y_,
                        active_users=active_y_indices,
                        A=A,
                        alpha_max=alpha_max,
                        alpha_min=alpha_min,
                        eta_user=conf.eta_male,
                        eta_item=conf.eta_female,                        
                    )

                rel_matrix = rel_vec_true_t * expo_alloc
                results[method]["exposure_x"][t] = (results[method]["exposure_x"][t - 1] * (t - 1)) / t
                results[method]["exposure_y"][t] = (results[method]["exposure_y"][t - 1] * (t - 1) + expo_alloc) / t
                results[method]["fair_y"][t] = calc_fair(match=merit_y, exposure=results[method]["exposure_y"][t])
                results[method]["fair_x"][t] = results[method]["fair_x"][t - 1]

                if reward_type == "n_match":
                    results[method]["match_x"][t] = results[method]["match_x"][t - 1].copy()
                    results[method]["match_x"][t][x_t] += rel_matrix.sum()
                elif reward_type == "match_prob":
                    results[method]["match_x"][t][x_t] = rel_matrix.sum() / expo_alloc.sum() / K
                elif reward_type == "n_match_per":
                    results[method]["match_x"][t] = (results[method]["match_x"][t - 1] * (t - 1)) / t
                    results[method]["match_x"][t][x_t] = (
                        results[method]["match_x"][t - 1][x_t] * (t - 1) + rel_matrix.sum()
                    ) / t

                if method == "MRet":
                    results[method]["user_retain_x"][t] = np.clip(
                        model.predict(
                            np.concatenate(
                                [x, results[method]["match_x"][t].reshape(-1, 1)],
                                axis=1,
                            ),
                            method,
                        ),
                        0,
                        1,
                    )

                if reward_type == "n_match":
                    results[method]["match_y"][t] = results[method]["match_y"][t - 1] + rel_matrix
                elif reward_type == "match_prob":
                    results[method]["match_y"][t] = rel_matrix / expo_alloc
                elif reward_type == "n_match_per":
                    results[method]["match_y"][t] = (
                        results[method]["match_y"][t - 1] * (t - 1) + rel_matrix
                    ) / t

                if method == "MRet":
                    results[method]["user_retain_y"][t] = np.clip(
                        model.predict(
                            np.concatenate(
                                [y, results[method]["match_y"][t].reshape(-1, 1)],
                                axis=1,
                            ),
                            method,
                        ),
                        0,
                        1,
                    )
            else:
                e = exam_func(n_x, K, shape=ranking_metric)
                y_t = random_.choice(active_y_indices)
                rel_vec_true_t = rel_mat_true[:, y_t]
                rel_vec_obs_t = rel_mat_obs[:, y_t]
                merit = merit_x
                A = np.sum(e[:K])
                alpha_max = np.max(e[:K])
                alpha_min = np.min(e[:K])


                if reward_type == "n_match":
                    match_y = results[method]["match_y"][t - 1][y_t] + A * rel_vec_obs_t
                    match_x = results[method]["match_x"][t - 1] + alpha_max * rel_vec_obs_t
                elif reward_type == "n_match_per":
                    match_y = (results[method]["match_y"][t - 1][y_t] * (t - 1) + A * rel_vec_obs_t) / t
                    match_x = (results[method]["match_x"][t - 1] * (t - 1) + alpha_max * rel_vec_obs_t) / t

                if method == "Uniform":
                    expo_alloc = uniform_ranking(rel_vec_obs_t, e, active_x_indices, random_)
                elif method == "MaxMatch":
                    expo_alloc = naive_ranking(rel_vec_obs_t, e, active_x_indices)
                elif method == "FairCo":
                    expo_alloc = fairco_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.1,
                    )
                elif method == "FairCo (lam=0.01)":
                    expo_alloc = fairco_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.01,
                    )
                elif method == "FairCo (lam=0.1)":
                    expo_alloc = fairco_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.1,
                    )
                elif method == "FairCo (lam=100)":
                    expo_alloc = fairco_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=100,
                    )
                elif method == "FairCo (equal exposure) (lam=0.1)":
                    expo_alloc = expfair_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=0.1,
                    )
                elif method == "FairCo (equal exposure) (lam=1)":
                    expo_alloc = expfair_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=1,
                    )
                elif method == "FairCo (equal exposure) (lam=10)":
                    expo_alloc = expfair_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=10,
                    )
                elif method == "FairCo (equal exposure) (lam=100)":
                    expo_alloc = expfair_ranking(
                        active_x_indices,
                        merit=merit,
                        rel_vec_obs_t=rel_vec_obs_t,
                        exposure=results[method]["exposure_x"][t - 1],
                        tau=t,
                        e=e,
                        lam=100,
                    )
                elif method == "MRet":
                    u_y_, u_x_ = predict_match(
                        user=y[y_t],
                        item=x,
                        m_user=results[method]["match_y"][t - 1][y_t],
                        m_item=results[method]["match_x"][t - 1],
                        r_hat=rel_vec_obs_t,
                        r=rel_vec_obs_t,
                        A=A,
                        alpha_max=alpha_max,
                        model=model,
                        method=method,
                    )
                    expo_alloc = user_retain_ranking(
                        e=e,
                        u_user=results[method]["user_retain_y"][t - 1][y_t],
                        u_item=results[method]["user_retain_x"][t - 1],
                        u_user_=u_y_,
                        u_item_=u_x_,
                        active_users=active_x_indices,
                        A=A,
                        alpha_max=alpha_max,
                        alpha_min=alpha_min,
                        eta_user=conf.eta_female,
                        eta_item=conf.eta_male,                        
                    )
                elif method == "MRet (noise)":
                    true_u_y_ = user_retain_func(alpha_y[y_t], beta_y[y_t], match_y)
                    true_u_x_ = user_retain_func(alpha_x.reshape(-1), beta_x.reshape(-1), match_x)
                    expo_alloc = user_retain_ranking(
                        e=e,
                        u_user=results[method]["true_user_retain_y"][t - 1][y_t],
                        u_item=results[method]["true_user_retain_x"][t - 1],
                        u_user_=random_.normal(true_u_y_, noise, size=true_u_y_.shape),
                        u_item_=random_.normal(true_u_x_, noise, size=true_u_x_.shape),
                        active_users=active_x_indices,
                        A=A,
                        alpha_max=alpha_max,
                        alpha_min=alpha_min,
                        eta_user=conf.eta_female,
                        eta_item=conf.eta_male,                         
                    )
                elif method == "MRet (best)":
                    true_u_y_ = user_retain_func(alpha_y[y_t], beta_y[y_t], match_y)
                    true_u_x_ = user_retain_func(alpha_x.reshape(-1), beta_x.reshape(-1), match_x)
                    expo_alloc = user_retain_ranking(
                        e=e,
                        u_user=results[method]["true_user_retain_y"][t - 1][y_t],
                        u_item=results[method]["true_user_retain_x"][t - 1],
                        u_user_=true_u_y_,
                        u_item_=true_u_x_,
                        active_users=active_x_indices,
                        A=A,
                        alpha_max=alpha_max,
                        alpha_min=alpha_min,
                        eta_user=conf.eta_female,
                        eta_item=conf.eta_male,                         
                    )
                elif method == "optimal_ranking":
                    expo_alloc = optimal_ranking(
                        e=e,
                        alpha_user=alpha_y[y_t],
                        beta_user=beta_y[y_t],
                        alpha_item=alpha_x.reshape(-1),
                        beta_item=beta_x.reshape(-1),
                        m_user=results[method]["match_y"][t - 1][y_t],
                        m_item=results[method]["match_x"][t - 1],
                        rel_vec_true_t=rel_vec_true_t,
                        active_users=active_x_indices,
                        K=K,
                    )

                rel_matrix = rel_vec_true_t * expo_alloc
                results[method]["exposure_y"][t] = (results[method]["exposure_y"][t - 1] * (t - 1)) / t
                results[method]["exposure_x"][t] = (results[method]["exposure_x"][t - 1] * (t - 1) + expo_alloc) / t
                results[method]["fair_x"][t] = calc_fair(match=merit_x, exposure=results[method]["exposure_x"][t])
                results[method]["fair_y"][t] = results[method]["fair_y"][t - 1]

                if reward_type == "n_match":
                    results[method]["match_y"][t] = results[method]["match_y"][t - 1].copy()
                    results[method]["match_y"][t][y_t] += rel_matrix.sum()
                elif reward_type == "match_prob":
                    results[method]["match_y"][t][y_t] = rel_matrix.sum() / expo_alloc.sum() / K
                elif reward_type == "n_match_per":
                    results[method]["match_y"][t] = (results[method]["match_y"][t - 1] * (t - 1)) / t
                    results[method]["match_y"][t][y_t] = (
                        results[method]["match_y"][t - 1][y_t] * (t - 1) + rel_matrix.sum()
                    ) / t

                if method == "MRet":
                    results[method]["user_retain_y"][t] = np.clip(
                        model.predict(
                            np.concatenate(
                                [y, results[method]["match_y"][t].reshape(-1, 1)],
                                axis=1,
                            ),
                            method,
                        ),
                        0,
                        1,
                    )

                if reward_type == "n_match":
                    results[method]["match_x"][t] = results[method]["match_x"][t - 1] + rel_matrix
                elif reward_type == "match_prob":
                    results[method]["match_x"][t] = (results[method]["match_x"][t - 1] + rel_vec_true_t) / 2
                elif reward_type == "n_match_per":
                    results[method]["match_x"][t] = (
                        results[method]["match_x"][t - 1] * (t - 1) + rel_matrix
                    ) / t

                if method == "MRet":
                    results[method]["user_retain_x"][t] = np.clip(
                        model.predict(
                            np.concatenate(
                                [x, results[method]["match_x"][t].reshape(-1, 1)],
                                axis=1,
                            ),
                            method,
                        ),
                        0,
                        1,
                    )

            results[method]["true_user_retain_x"][t] = user_retain_func(
                alpha_x_.reshape(-1),
                beta_x_.reshape(-1),
                results[method]["match_x"][t],
            )
            results[method]["true_user_retain_y"][t] = user_retain_func(
                alpha_y_.reshape(-1),
                beta_y_.reshape(-1),
                results[method]["match_y"][t],
            )

            user_retain_mask_x = random_.rand(n_x) < results[method]["true_user_retain_x"][t]
            user_retain_mask_y = random_.rand(n_y) < results[method]["true_user_retain_y"][t]

            diff_users_x = random_.rand(n_x) < candidate_retention * results[method]["active_users_x"][t - 1].sum() / n_x
            diff_users_y = random_.rand(n_y) < candidate_retention * results[method]["active_users_y"][t - 1].sum() / n_y
            diff_users_x = results[method]["active_users_x"][t - 1].astype(bool) * diff_users_x
            diff_users_y = results[method]["active_users_y"][t - 1].astype(bool) * diff_users_y

            results[method]["active_users_x"][t] = results[method]["active_users_x"][t - 1].copy()
            results[method]["active_users_y"][t] = results[method]["active_users_y"][t - 1].copy()
            results[method]["active_users_x"][t][diff_users_x] = (
                user_retain_mask_x[diff_users_x] * results[method]["active_users_x"][t - 1][diff_users_x]
            )
            results[method]["active_users_y"][t][diff_users_y] = (
                user_retain_mask_y[diff_users_y] * results[method]["active_users_y"][t - 1][diff_users_y]
            )
