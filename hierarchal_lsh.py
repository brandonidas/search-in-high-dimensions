import numpy as np

class HierarchicalLSH:
    def __init__(self, d, num_levels, num_tables, num_buckets):
        self.d = d  # Dimensionality of the vectors
        self.num_levels = num_levels  # Number of levels
        self.num_tables = num_tables  # Number of hash tables per level
        self.num_buckets = num_buckets  # Number of buckets per table
        self.levels = []

        # Initialize hash tables and hash functions at each level
        for level in range(num_levels):
            tables = []
            hash_functions = []
            for _ in range(num_tables):
                table = [[] for _ in range(num_buckets)]
                tables.append(table)
                hash_func = self._generate_hash_function()
                hash_functions.append(hash_func)
            self.levels.append((tables, hash_functions))

    def _generate_hash_function(self):
        # Generate a random projection vector for the hash function
        return np.random.randn(self.d)

    def _hash(self, vector, hash_func):
        # Compute the hash value for a vector using a hash function
        return hash(np.dot(vector, hash_func) >= 0)

    def index(self, dataset):
        # Index the dataset by hashing vectors into tables at each level
        for vector in dataset:
            for level in self.levels:
                tables, hash_functions = level
                for table, hash_func in zip(tables, hash_functions):
                    hash_value = self._hash(vector, hash_func)
                    table[hash_value].append(vector)

    def query(self, query_vector):
        # Find the most similar vector to the query using hierarchical LSH
        max_inner_product = float('-inf')
        most_similar_vector = None

        for level in self.levels:
            tables, hash_functions = level
            level_candidates = []
            for table, hash_func in zip(tables, hash_functions):
                hash_value = self._hash(query_vector, hash_func)
                bucket = table[hash_value]
                level_candidates.extend(bucket)

            for vector in level_candidates:
                inner_product = np.dot(vector, query_vector)
                if inner_product > max_inner_product:
                    max_inner_product = inner_product
                    most_similar_vector = vector
        
        return most_similar_vector, max_inner_product
