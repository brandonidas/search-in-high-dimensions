import numpy as np
from bandit_mips import BanditMIPS

def compute_ground_truth(query, vectors):
  similarities = np.dot(vectors, query)
  most_similar_index = np.argmax(similarities)
  most_similar_vector = vectors[most_similar_index]
  return most_similar_vector

def test_BanditMIPS():
    # Test case 1
    num_rows = 1000 # vectors
    d = 1000  # Dimension
    np.random.seed(42)

    mean = 1 # BANDIT MIPS DELTA only works with positive values in my implementation
    # despite claioms form the paper on being able to process non-negative values
    std_dev = 1
    v = np.random.normal(loc=mean, scale=std_dev, size=(num_rows, d))

    # Pop the query vector from v
    query = v[0]
    v = v[1:, :]

    delta = 0.1
    a = mean - 2 * std_dev
    b = mean + 2 * std_dev
    variance_proxy = np.ones(v.shape[0]) * (b ** 2 - a**2)/4
    output, op_count = BanditMIPS(v, query, delta, variance_proxy) 
    dot_product = np.dot(v[output], query)
    if output != None:
        print("MIP: " + str(dot_product))
    ground_truth_vector = compute_ground_truth(query, v)
    print("Ground truth: " + str(np.dot(ground_truth_vector, query)))

    print("op_count:" + str(op_count) + " which is " 
          + str(op_count / (num_rows * d) * 100) + "% of the naive method" )

def test_BanditMIPS_varying_column_distribution():
    # Test case 1
    num_rows = 1100 # vectors
    d = 1000  # Dimension
    np.random.seed(42)
    
    mean = 0
    std_dev = 1
    v = np.random.normal(loc=mean, scale=std_dev, size=(num_rows, d)).T

    # Pop the query vector from v
    query = v[0]
    v = v[1:, :]

    delta = 0.1
    variance_proxy = std_dev * std_dev * np.ones(v.shape[0]) # just for this case because it is standard normal, variance = 1
    output = BanditMIPS(v, query, delta, variance_proxy) 
    dot_product = np.dot(v[output], query)
    if output != None:
        print("MIP: " + str(dot_product))
    ground_truth_vector = compute_ground_truth(query, v)
    print("Ground truth: " + str(np.dot(ground_truth_vector, query)))


# Run the test function
test_BanditMIPS()
# test_BanditMIPS_varying_column_distribution()