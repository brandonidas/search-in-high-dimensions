import numpy as np
from mips_lsh import LSH
from hierarchal_lsh import HierarchicalLSH


# Example usage
n = 10000
d = 1000

mean = 0
std_dev = 1

# Generate random data with a normal distribution
query = np.random.normal(mean, std_dev, size=(d,))

rate = 0.5
# query = np.random.exponential(scale=1/rate, size=(d,))

dataset = []
for _ in range(n):
    # new_vector = np.random.normal(mean, std_dev, size=(d,))
    # Generate random data with an exponential distribution instead of normal in order to increase disimilarity.
    # TODO replicate code such that the probability function is changeable
    new_vector = np.random.exponential(scale=1/rate, size=(d,))
    dataset.append(new_vector)
print('Dataset generated')
# Create LSH object and index the dataset
lsh = LSH(d, num_tables=2, num_buckets=2)
lsh.index(dataset)

# Query the LSH for the most similar vector to the query
base_lsh_most_similar_vector, similarity_score = lsh.query(query)

# print("Most similar vector:", most_similar_vector)
print("Base LSH Similarity score:", similarity_score)

hierarchal_lsh = HierarchicalLSH(d, num_levels=2, num_tables=2, num_buckets=2)
hierarchal_lsh.index(dataset)
_, similarity_score = hierarchal_lsh.query(query)
print("Hierarchal LSH similarity score:", similarity_score)

# true exhaustive search
# Compute inner products between query and dataset vectors
inner_products = np.dot(dataset, query)

# Find the vector with the highest inner product
most_similar_vector_index = np.argmax(inner_products)
most_similar_vector_inner_product = inner_products[most_similar_vector_index]
print("True max inner product:" + str(most_similar_vector_inner_product))