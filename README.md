## Beyond Match Maximization and Fairness: Retention-Objectified Two-Sided Matching

This repository contains the code to replicate experiments conducted in the paper "[Beyond Match Maximization and Fairness: Retention-Objectified Two-Sided Matching](https://openreview.net/forum?id=g2cZaKmRrc)" by Ren Kishimoto, Rikiya Takehi, Koichi Tanaka, Yoji Tomita, Masahiro Nomura, Riku Togashi, and Yuta Saito, which has been accepted to ICLR2026.



## Abstract
On two-sided matching platforms such as online dating and recruiting, recommendation algorithms often aim to maximize the total number of matches. However, this objective creates an imbalance, where some users receive far too many matches while many others receive very few and eventually abandon the platform. Retaining users is crucial for many platforms, such as those that depend heavily on subscriptions. Some may use fairness objectives to solve the problem of match maximization. However, fairness in itself is not the ultimate objective for many platforms, as users do not suddenly reward the platform simply because exposure is equalized. In practice, where user retention is often the ultimate goal, casually relying on fairness will leave the optimization of retention up to luck.
In this work, instead of maximizing matches or axiomatically defining fairness, we formally define the new problem setting of maximizing user retention in two-sided matching platforms. To this end, we introduce a dynamic learning-to-rank (LTR) algorithm called **M**atching for **Ret**ention (**MRet**). Unlike conventional algorithms for two-sided matching, our approach models user retention by learning personalized retention curves from each user’s profile and interaction history. Based on these curves, MRet dynamically adapts recommendations by jointly considering the retention gains of both the user receiving recommendations and those who are being recommended, so that limited matching opportunities can be allocated where they most improve overall retention. Naturally but importantly, empirical evaluations on synthetic and real-world datasets from a major online dating platform show that MRet achieves higher user retention, since conventional methods optimize matches or fairness rather than retention.

## Citation

```
@inproceedings{
kishimoto2026beyond,
title={Beyond Match Maximization and Fairness: Retention-Objectified Two-Sided Matching},
author={Ren Kishimoto and Rikiya Takehi and Koichi Tanaka and Yoji Tomita and Masahiro Nomura and Riku Togashi and Yuta Saito},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=g2cZaKmRrc}
}
```

## Dependencies
This repository supports Python 3.10.6 or newer.

- numpy==1.23.5
- pandas==1.5.2
- scikit-learn==1.1.3
- matplotlib==3.7.1
- torch==1.12.0


## Running the Code
The commands needed to reproduce the experiments are summarized below. Please move under the `synthetic` directly first and then run the notebooks.

### Synthetic Data

```bash
# How does MRet perform as the timestep $t$ increases?
synthetic/main/main_seed.ipynb
synthetic/main/show.ipynb

# How does MRet perform when user popularity varies?
synthetic/main/main_kappa.ipynb
synthetic/main/show.ipynb

# Why does Fairco underperform in user retention?
synthetic/main/main_seed.ipynb
synthetic/main/show.ipynb

# How does MRet perform when varying the number of users?
synthetic/main/main_n_xy.ipynb
synthetic/main/show.ipynb

# How does the hyperparameter of FairCo affect its performance?
synthetic/main/main_lambda.ipynb
synthetic/main/show.ipynb

# How does MRet perform under varying noise levels in the match probabilities?
synthetic/main/main_noise.ipynb
synthetic/main/show.ipynb

# How does MRet perform when popularity drifts over time? 
synthetic/main/main_popularity.ipynb
synthetic/main/show.ipynb

# How does FairCo perform under the equal-exposure fairness criterion? 
synthetic/main/main_seed.ipynb
synthetic/main/show.ipynb

# How accurate is MRet as an approximation to the Optimal method?
synthetic/main/main_seed_optimal.ipynb
synthetic/main/show.ipynb

```
