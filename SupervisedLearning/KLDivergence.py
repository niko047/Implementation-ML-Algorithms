import torch
import numpy as np
import pandas as pd


n = torch.distributions.normal.Normal(loc=0, scale=4)
u = torch.distributions.uniform.Uniform(low=-2.5, high=2.5)

samples_n = n.sample(sample_shape=(10000,))
samples_u = u.sample(sample_shape=(10000,))

# Task is to compute the KL divergence between those two distributions
# 1. Sample a bunch of points from a distribution
# 2. Batch them in bins of adequate length
# 3. For every bin calculate the respective P(xi) * P(xi)/Q(xi)

def kl_divergence(p, q):
    """Computes the KL divergence between the two samples from two distributions"""

    # Get the smallest value of the two arrays and the largest
    max_1, min_1 = p.max(), p.min()
    max_2, min_2 = q.max(), q.min()

    try:
        assert max_1 > max_2 and min_1 < min_2
    except AssertionError:
        print(f"Support of Q must be fully included in the support of P.")

    # Create a linspaced array of bins
    bins = np.linspace(start=min_1, stop=max_1, num=150, endpoint=True)

    dist_1 = pd.Series(np.digitize(p, bins, right=True)).value_counts().sort_index()

    normalized_dist_1 = dist_1/dist_1.sum()

    dist_2 = pd.Series(np.digitize(q, bins, right=True)).value_counts().sort_index()

    normalized_dist_2 = dist_2/dist_2.sum()
    KL_distance = 0

    for _p, _q in zip(normalized_dist_1, normalized_dist_2):
        KL_distance += _p * np.log(_q/_p)

    print(f"The Kullback Leibler divergence is {KL_distance}")

if __name__ == '__main__':
    kl_divergence(samples_n, samples_u)
