import math
import numpy as np
from boundedME import BoundedME

def compute_ground_truth(query_vector, A_arms, K = 1):
    dot_products = np.dot(A_arms, query_vector)
    max_index = np.argmax(dot_products)
    
    return A_arms[max_index]

def test_bounded_ME():
    # Test case
    np.random.seed(42)
    A_arms = np.random.normal(size=(1000, 1000))
    query_vector = np.random.normal(size=1000)
    epsilon = 0.1 # suboptimality
    delta = 0.1 # probability margin # TODO seems to have no effect
    K = 1
    
    S = BoundedME(epsilon, delta, A_arms, query_vector, K)
    
    # Use the last level in S
    S_last = S[0]
    
    # Compute the ground truth result
    ground_truth_vector = compute_ground_truth(query_vector, A_arms)
    
    # Compare results
    print(type(S))
    #dot_product_S_last = np.dot(S, query_vector)
    dot_product_ground_truth = np.dot(ground_truth_vector, query_vector)
    
    #print("BoundedME Result (Last Level):", dot_product_S_last)
    print("Ground Truth Result:", dot_product_ground_truth)
#test_bounded_ME()

def test_bounded_ME_with_Tracking():
    # Test case
    np.random.seed(42)
    A_arms = np.random.normal(size=(1000, 1000))
    query_vector = np.random.normal(size=1000)
    epsilon = 0.1 # suboptimality
    delta = 0.1 # probability margin # TODO seems to have no effect
    K = 1
    
    S, count = BoundedME(epsilon, delta, A_arms, query_vector, K)
    
    # Use the last level in S
    S_last = S[-1]
    
    # Compute the ground truth result
    ground_truth_vector = compute_ground_truth(query_vector, A_arms)
    
    # Compare results
    dot_product_S_last = np.dot(S, query_vector)
    dot_product_ground_truth = np.dot(ground_truth_vector, query_vector)
    
    print("BoundedME Result (Last Level):", dot_product_S_last)
    print("Ground Truth Result:", dot_product_ground_truth)
    
    print("OpCount: " + str(count))
    print("vs " + str(A_arms.shape[0] * A_arms.shape[1]) + "... which is empirically "\
          + str( math.ceil(100*count / (A_arms.shape[0] * A_arms.shape[1]))) + "% of brute force")
    
#test_bounded_ME_with_Tracking()

def test_bounded_ME_with_Tracking_on_columns_with_distribution():
    v, query_vector = set_up_atoms_and_query()
    epsilon = 0.1 # suboptimality
    delta = 0.3 # probability margin # TODO seems to have no effect
    K = 10 

    S, count = BoundedME(epsilon, delta, v, query_vector, K)
    
    # Use the last level in S
    
    # Compute the ground truth result
    ground_truth_vector = compute_ground_truth(query_vector, v)
    
    # Compare results
    avg_dot_product_S_last = np.average(np.dot(S, query_vector))
    dot_product_ground_truth = np.dot(ground_truth_vector, query_vector)
    
    print("BoundedME Result (Last Level):", avg_dot_product_S_last)
    print("Ground Truth Result:", dot_product_ground_truth)
    
    print("OpCount: " + str(count))
    print("vs " + str(v.shape[0] * v.shape[1]) + "... which is empirically "\
          + str(math.ceil(100*count / (v.shape[0] * v.shape[1]))) + "% of brute force")


def test_bounded_ME_with_Tracking_on_columns_with_diverse_distribution():
    v, query_vector = set_up_atoms_and_query()
    epsilon = 0.3 # suboptimality
    delta = 0.3 # probability margin # TODO seems to have no effect
    K = 10 

    S, count = BoundedME(epsilon, delta, v, query_vector, K)
    
    # Use the last level in S
    
    # Compute the ground truth result
    ground_truth_vector = compute_ground_truth(query_vector, v)
    
    # Compare results
    avg_dot_product_S_last = np.average(np.dot(S, query_vector))
    dot_product_ground_truth = np.dot(ground_truth_vector, query_vector)
    
    print("BoundedME Result (Last Level):", avg_dot_product_S_last)
    print("Ground Truth Result:", dot_product_ground_truth)
    
    print("OpCount: " + str(count))
    print("vs " + str(v.shape[0] * v.shape[1]) + "... which is empirically "\
          + str(math.ceil(100*count / (v.shape[0] * v.shape[1]))) + "% of brute force")


def generate_matrix_with_different_distributions(num_rows, num_columns):
    matrix = np.empty((num_rows, num_columns))
    
    for current_dim in range(num_columns):
        distribution = np.random.choice(['uniform', 'normal', 'exponential', 'bernoulli', 'beta'])
        
        if distribution == 'uniform':
            low = np.random.uniform(-10, 10)
            high = np.random.uniform(low, 10)
            column = np.random.uniform(low, high, size=num_rows)
        
        elif distribution == 'normal':
            mean = np.random.uniform(-10, 10)
            std_dev = np.random.uniform(0.1, 10)
            column = np.random.normal(loc=mean, scale=std_dev, size=num_rows)
        
        elif distribution == 'exponential':
            scale = np.random.uniform(0.1, 10)
            column = np.random.exponential(scale=scale, size=num_rows)
        
        elif distribution == 'bernoulli':
            p = np.random.uniform(0, 1)
            column = np.random.binomial(n=1, p=p, size=num_rows)
        
        elif distribution == 'beta':
            a = np.random.uniform(0.1, 5)
            b = np.random.uniform(0.1, 5)
            column = np.random.beta(a=a, b=b, size=num_rows)
        
        column[column < 0] = 0
        matrix[:, current_dim] = column
    
    return matrix


def set_up_atoms_and_query():
    num_rows = 10000 # vectors
    d = 1000  # Dimension
    np.random.seed(42)
    mean = np.random.uniform(1, 10)
    std_dev = np.random.uniform(1, 10)
    # v = np.random.normal(loc=mean, scale=std_dev, size=(num_rows, d)).T
    v = generate_matrix_with_different_distributions(num_rows, d)

    # Pop the query vector from v
    query_vector = v[0]
    v = v[1:, :]
    return v,query_vector

def generate_matrix(num_rows, num_columns):
    matrix = np.empty((num_rows, num_columns))
    variances = np.empty(num_columns)
    
    for current_dim in range(num_columns):
        mean = np.random.uniform(1, 10)
        std_dev = np.random.uniform(1, 10)
        column = np.random.normal(loc=mean, scale=std_dev, size=num_rows)
        matrix[:, current_dim] = column
        variances[current_dim] = np.var(column)
    
    return matrix, variances
 
test_bounded_ME_with_Tracking_on_columns_with_distribution()

