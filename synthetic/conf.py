num_seeds = 1
T = 2000
n_x = 1000
n_y = 1000
n_xy_list = [250, 500, 1000, 2000, 4000]

n_train = 5000
K = 5

dim = 10
rel_noise = 0
rel_noise_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

noise = 0.5
proportion = 0.5
proportion_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

kappa = 0.5
kappa_list = [0, 0.25, 0.5, 0.75, 1.0]

reward_shape = "semi_quadratic"
reward_type = "n_match"

if reward_type == "n_match":
    alpha_param = [2, 1]

beta_param = [0.5, 0.2]

candidate_retention = 0.002

ranking_metric = "inv"
lambda_ = 0.1
lambda_list = [0.01, 0.1, 1, 10, 100]

eta_female = 0.5
eta_male = 0.5
eta_female_list = [0, 0.25, 0.5, 0.75, 1.0]
eta_male_list = [1.0, 0.75, 0.5, 0.25, 0]


method_list = ['Uniform', "MaxMatch", "FairCo (lam=100)", 'MRet','FairCo (equal exposure) (lam=100)']

show_method_list = method_list

color_dict = {
    "MRet": "tab:red",
    "MRet (best)": "tab:blue",
    "Uniform": "tab:green",
    "MaxMatch": "tab:cyan",
    "FairCo": "tab:pink",
    "optimal_ranking": "tab:orange",
    "FairCo (lam=0.01)": "tab:purple",
    "FairCo (lam=0.1)": "tab:olive",
    "FairCo (lam=100)": "tab:pink",
    "MRet (noise)": "brown",
    "FairCo (equal exposure) (lam=0.1)": "blueviolet",
    "FairCo (equal exposure) (lam=1)": "royalblue",
    "FairCo (equal exposure) (lam=10)": "teal",
    "FairCo (equal exposure) (lam=100)": "blueviolet",
    
}

relative = None
random_state = 12345

time_popularity = False
