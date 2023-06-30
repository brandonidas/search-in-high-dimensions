import numpy as np

class LSH:
    def __init__(self, d, num_tables, num_buckets):
        self.d = d  # Dimensionality of the vectors
        self.num_tables = num_tables  # Number of hash tables
        self.num_buckets = num_buckets  # Number of buckets per table
        self.tables = []
        self.hash_functions = []

        # Initialize hash tables and hash functions
        for _ in range(num_tables):
            table = [[] for _ in range(num_buckets)]
            self.tables.append(table)
            hash_func = self._generate_hash_function()
            self.hash_functions.append(hash_func)

    def _generate_hash_function(self):
        # Generate a random projection vector for the hash function
        return np.random.randn(self.d)

    def _hash(self, vector, hash_func):
        # Compute the hash value for a vector using a hash function
        return hash(np.dot(vector, hash_func) >= 0)

    def index(self, dataset):
        # Index the dataset by hashing vectors into tables
        for vector in dataset:
            for table, hash_func in zip(self.tables, self.hash_functions):
                hash_value = self._hash(vector, hash_func)
                table[hash_value].append(vector)

    def query(self, query_vector):
        # Find the most similar vector to the query using LSH
        max_inner_product = float('-inf')
        most_similar_vector = None

        for table, hash_func in zip(self.tables, self.hash_functions):
            hash_value = self._hash(query_vector, hash_func)
            bucket = table[hash_value]
            
            for vector in bucket:
                inner_product = np.dot(vector, query_vector)
                if inner_product > max_inner_product:
                    max_inner_product = inner_product
                    most_similar_vector = vector
        
        return most_similar_vector, max_inner_product