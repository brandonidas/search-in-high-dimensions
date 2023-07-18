import torch
import torch.nn.functional as F
import math
import numpy as np

def compute_m(delta, epsilon, A_arms):
    b = torch.max(A_arms, dim=0).values
    a = torch.zeros_like(b)  # torch.min(A_arms, dim=0).values

    u = ((np.log(1/delta) / 2) * ((b - a) ** 2) / (epsilon ** 2))[0]
    m = min((u + 1) / (1 + len(A_arms) * u), (u + u) / (1 + len(A_arms) * u))
    return m


def pull_number_modifier(current_epsilon, current_delta, S_size, K):
    epsilon_coefficient = 2 / (current_epsilon ** 2)
    log_term_numerator = 2 * (S_size - K)
    log_term_denominator = current_delta * ((S_size - K) / 2 + 1)
    return epsilon_coefficient * math.log(log_term_numerator / log_term_denominator)

def remove_low_means(S, mean_dict, K):
    threshold_margin = math.ceil((len(S) - K) / 2)
    sorted_mean_dict = sorted(mean_dict.items(), key=lambda x: x[1], reverse=True)

    threshold_mean = sorted_mean_dict[threshold_margin - 1][1]
    new_mean_dict = dict(filter(lambda x: x[1] >= threshold_mean, mean_dict.items()))
    return new_mean_dict

# Bounded Median Estimator
def BoundedME( A_arms, query_vector, K, epsilon=0.1, delta=0.3):
    # For tracking floating point operations count
    op_count = 0

    S_current = np.arange(len(A_arms))
    current_epsilon, current_delta = epsilon / 4, delta / 2
    reward_dictionary = {i: 0 for i in S_current}

    m = compute_m(delta, epsilon, A_arms)
    old_t = 0
    while len(S_current) > K:
        new_t = math.floor(m * pull_number_modifier(current_epsilon, current_delta, len(S_current), K))
        op_count += 6

        reward_dictionary, ops = pull_new_arms(query_vector, A_arms, K, S_current, reward_dictionary, old_t, new_t)
        op_count += ops

        S_current = np.array([i for i, _ in reward_dictionary.items()])
        current_epsilon = current_epsilon * 3/4
        current_delta = current_delta / 2
        old_t = new_t
        op_count += 3

    return S_current, op_count

def pull_new_arms(query_vector, A_arms, K, S_current, reward_dictionary, old_t, new_t):
    ops = 0
    for i in S_current:
        a = A_arms[i]
        #print(torch.tensor(a)[old_t:new_t])
        arm_pulls = torch.dot(query_vector[old_t:new_t], a.clone().detach().requires_grad_(True)[old_t:new_t])
        reward = arm_pulls  # / (new_t - old_t - 1) # no need for actual mean
        ops += query_vector[old_t:new_t].numel()  # Counting multiplications inside torch.dot

        reward_dictionary[i] += reward

    reward_dictionary = remove_low_means(S_current, reward_dictionary, K)
    ops += 2
    return reward_dictionary, ops


import math
import torch
import torch.distributions as dist
import numpy as np

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

# test_bounded_ME_with_Tracking_on_columns_with_distribution()

import torch
import torch.nn.functional as F

def approximate_matrix_multiplication(A, B, K, epsilon=0.1, delta=0.3):
    assert A.size(1) == B.size(0), "Matrix dimensions are incompatible for multiplication"

    m, n = A.size(0), B.size(1)
    p = A.size(1)

    # Initialize the output matrix
    C = torch.zeros((m, n))

    for i in range(m):
        query_vector = A[i]
        S_current, _ = BoundedME(B.T, query_vector, K, epsilon, delta)
        
        for j in S_current:
            C[i, j] = torch.dot(query_vector, B.T[j])

    return C

def compute_ground_truth(A, B, K):
    result = torch.matmul(A, B)
    _, indices = torch.topk(result, K, dim=1)
    topk_result = torch.zeros_like(result).scatter_(1, indices, result.gather(1, indices))
    return topk_result

def test_approximate_matrix_multiplication():
    # Test case
    torch.manual_seed(42)
    A = torch.randn(100, 50)
    B = torch.randn(50, 200)
    K = 30
    epsilon = 0.01
    delta = 0.3

    # Compute the ground truth result
    ground_truth_result = compute_ground_truth(A, B, K)

    # Compute the approximate result using BoundedME
    approximate_result = approximate_matrix_multiplication(A, B, K, epsilon, delta)

    # Compare results
    error = torch.norm(approximate_result - ground_truth_result)
    print("Error:", error)
    print(approximate_result)
    print("VS")
    print(ground_truth_result)

    # Compute accuracy
    total_elements = ground_truth_result.numel()
    correct_elements = torch.sum(approximate_result == ground_truth_result)
    accuracy = (correct_elements / total_elements) * 100
    print("Accuracy: {:.2f}%".format(accuracy))

test_approximate_matrix_multiplication()