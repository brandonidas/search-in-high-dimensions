import math
import torch
import torch.distributions as dist
import numpy as np
from torchBME import BME as BoundedME
from bandit_layer import BanditLayer

def compute_ground_truth(query_vector, A_arms, K=1):
    dot_products = torch.matmul(A_arms, query_vector)
    max_index = torch.argmax(dot_products)

    return max_index

def test_bounded_ME_with_Tracking_on_columns_with_distribution():
    v, query_vector = set_up_atoms_and_query()
    epsilon = 0.1  # suboptimality
    delta = 0.3  # probability margin # TODO seems to have no effect
    K = 10

    S, count = BoundedME(v, query_vector, K,epsilon, delta)

    # Use the last level in S

    # Compute the ground truth result
    ground_truth_index = compute_ground_truth(query_vector, v)

    # Compare results
    avg_dot_product_S_last = torch.mean(torch.matmul(v[S], query_vector))
    dot_product_ground_truth = torch.dot(v[ground_truth_index], query_vector)

    print("BoundedME Result (Last Level):", avg_dot_product_S_last)
    print("Ground Truth Result:", dot_product_ground_truth)

    print("OpCount: " + str(count))
    print("vs " + str(v.size(0) * v.size(1)) + "... which is empirically " + str(math.ceil(100 * count / (v.size(0) * v.size(1)))) + "% of brute force")


def generate_matrix_with_different_distributions(num_rows, num_columns):
    matrix = torch.empty((num_rows, num_columns))

    for current_dim in range(num_columns):
        distribution = torch.rand(1).item()

        if distribution < 0.2:
            low = torch.rand(1) * 20 - 10
            high = torch.rand(1) * 10 + low
            column = torch.rand(num_rows) * (high - low) + low

        elif distribution < 0.4:
            mean = torch.rand(1) * 20 - 10
            std_dev = torch.rand(1) * 10
            column = torch.randn(num_rows) * std_dev + mean

        elif distribution < 0.6:
            scale = torch.rand(1) * 10
            u = torch.rand(num_rows)
            column = -scale * np.log(1 - u)

        elif distribution < 0.8:
            p = torch.rand(1)
            column = dist.Bernoulli(p * torch.ones(num_rows)).sample()

        else:
            a = torch.rand(1) * 4 + 0.1
            b = torch.rand(1) * 4 + 0.1
            beta_dist = dist.Beta(a, b)
            column = beta_dist.sample((num_rows,))

        column[column < 0] = 0
        matrix[:, current_dim] = column.flatten()

    return matrix


def set_up_atoms_and_query():
    num_rows = 10000  # vectors
    d = 1000  # Dimension
    torch.manual_seed(42)
    mean = torch.rand(1) * 9 + 1
    std_dev = torch.rand(1) * 9 + 1
    v = generate_matrix_with_different_distributions(num_rows, d)

    # Pop the query vector from v
    query_vector = v[0, :]
    v = v[1:, :]
    return v, query_vector

test_bounded_ME_with_Tracking_on_columns_with_distribution()

def compute_ground_truth(query_vector, A_arms, K):
    dot_products = torch.matmul(A_arms, query_vector)
    _, indices = torch.topk(dot_products, K)
    return indices

def test_bounded_ME_with_Tracking_on_columns_with_distribution_and_K():
    v, query_vector = set_up_atoms_and_query()
    epsilon = 0.03  # suboptimality
    delta = 0.01  # probability margin # TODO seems to have no effect
    K = 3000
      # Updated value of K

    S, count = BoundedME(v, query_vector, K, epsilon, delta)

    # Compute the ground truth result for querying K vectors
    ground_truth_indices = compute_ground_truth(query_vector, v, K)

    S = torch.from_numpy(S)
    #ground_truth_indices = torch.from_numpy(ground_truth_indices)

    # Compare results
    avg_dot_product_S = torch.mean(torch.matmul(v[S], query_vector))
    dot_products_ground_truth = torch.matmul(v[ground_truth_indices], query_vector)

    print("BoundedME Result (Average Dot Products):", avg_dot_product_S)
    print("Ground Truth Result (Average Dot Products):", torch.mean(dot_products_ground_truth))

    # Compare the percentage of matching entries
    matching_entries = torch.sum(torch.isin(S, ground_truth_indices)).item()
    total_entries = K
    percentage_matching = (matching_entries / total_entries) * 100

    print("Matching Entries: {}/{}".format(matching_entries, total_entries))
    print("Percentage Matching: {:.2f}%".format(percentage_matching))
    print("Count from BME: {}".format(S.shape))

    print("OpCount:", count)
    print("vs", v.size(0) * v.size(1), "... which is empirically",
          math.ceil(100 * count / (v.size(0) * v.size(1))), "% of brute force")


test_bounded_ME_with_Tracking_on_columns_with_distribution_and_K()
