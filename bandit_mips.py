import math
import random
import numpy as np

# BanditMIPS https://arxiv.org/pdf/2212.07551.pdf

class BanditMIPS:
    def __init__(self, d, error_probability, sub_gaussian_parameter):
        self.d = d
        self.delta = error_probability
        self.sigma = sub_gaussian_parameter

    def banditSearcher(self, v, query):
        n = len(v)
        
        Potential_Solutions = list(range(1,n)) # assume notation means [0 through n-1]

        d_used = 0
        C_d_used = [np.inf] * n        
        max_normalised_inner_products_per_solution = [0]*n
            
        while d_used < self.d and len(Potential_Solutions) > 1:
            # J represents "pulling" a random arm in the multi-arm bandit
            J = random.randrange(1,self.d)
            for i in Potential_Solutions:
                # normalised inner product of v[i],
                max_normalised_inner_products_per_solution[i] = \
                    self.normalised_inner_product(v, query, d_used, i,\
                                                    max_normalised_inner_products_per_solution[i], J)
                # it is sufficient to find atom v with highest normalised inner product

                # construct confidence interval around (1 - delta/(2*n*d_{used}^2))
                C_d_used[i] = self.sigma * math.sqrt((2*math.log(4*n*d_used^2)/self.delta) \
                                                    / (d_used + 1))
                
            def solution_check(i):
                print(type(max_normalised_inner_products_per_solution[0]))
                i_prime = max_normalised_inner_products_per_solution.index(max(\
                    max_normalised_inner_products_per_solution))
                return max_normalised_inner_products_per_solution[i] + C_d_used[i] >= \
                max_normalised_inner_products_per_solution[i_prime] - C_d_used[i]

            Potential_Solutions = list(filter(lambda i: solution_check(i), Potential_Solutions))
            d_used =d_used + 1

        if len(Potential_Solutions) > 1:
            actual_inner_products = {}
            max_inner_product_sf, max_index_sf = 0,0
            for i in Potential_Solutions:
                actual_inner_products[i] = v[i] @ query
                if actual_inner_products[i] > max_inner_product_sf:
                    max_inner_product_sf = actual_inner_products[i]
                    max_index_sf = i
            return max_index_sf, v[max_index_sf]
        elif len(Potential_Solutions) == 1:
            return Potential_Solutions[0], v[Potential_Solutions[0]]
        else:
            print("Error: None Found")   

    def normalised_inner_product(self, v, query, d_used, i, mu_hat_i, J):
        return (d_used * mu_hat_i + v[i:J] * query[J]) \
                    / (d_used + 1)

def run_tests():
    bandit = BanditMIPS(d=10, error_probability=0.5, sub_gaussian_parameter=100)

    # Test 1: Basic Test with High Inner Product
    v = np.random.randn(100, 10)  # 100 candidate vectors of length 10
    query = np.random.randn(10)  # Query vector of length 10
    v[42] = query * 10
    index, vector = bandit.banditSearcher(v, query)
    inner_product = np.dot(vector, query)
    if inner_product > 9:  # Check if the inner product is higher than 9
        print("Test 1 Passed")
    else:
        print("Test 1 Failed")

    # Test 2: No High Inner Product Available
    v = np.random.randn(100, 10)  # 100 candidate vectors of length 10
    query = np.random.randn(10)  # Query vector of length 10
    index, vector = bandit.banditSearcher(v, query)
    inner_product = np.dot(vector, query)
    if inner_product < 0:  # Check if the inner product is negative (assuming a high inner product is positive)
        print("Test 2 Passed")
    else:
        print("Test 2 Failed")

    # Test 3: Multiple Vectors with High Inner Products
    v = np.random.randn(100, 10)  # 100 candidate vectors of length 10
    query = np.random.randn(10)  # Query vector of length 10
    v[25] = query * 5
    v[73] = query * 10
    index, vector = bandit.banditSearcher(v, query)
    inner_product = np.dot(vector, query)
    if inner_product > 9:  # Check if the inner product is higher than 9
        print("Test 3 Passed")
    else:
        print("Test 3 Failed")

run_tests()