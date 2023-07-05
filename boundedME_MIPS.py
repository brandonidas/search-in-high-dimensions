# Bounded Median Estimator
def BoundedMEwithTracking(epsilon, delta, A_arms, query_vector, K=1):
    # For tracking arithmetic operations count
    op_count = 0

    S = [None, A_arms]  # None at index 0 so we can have 1-based indexing
    epsil_array = [None, epsilon / 4]
    delta_array = [None, delta / 2]

    l = 1
    empirical_mean_dictionary = {tuple(a): 0 for a in A_arms}

    m = compute_m(delta, epsilon, A_arms)
    old_t = 0
    while len(S[l]) > K:
        new_t = math.floor(m * pull_number_modifier(epsil_array[l], delta_array[l], len(S[l]), K))
        op_count += 5

        for a in S[l]:
            arm_pulls = np.dot(query_vector[old_t:new_t], a[old_t:new_t])
            empirical_mean = arm_pulls / (new_t - old_t - 1)

            # Counting arithmetic operations
            op_count += query_vector[old_t:new_t].size  # Counting multiplications inside np.dot
            op_count += 1  # Division in empirical_mean calculation

            empirical_mean_dictionary[tuple(a)] += empirical_mean

        empirical_mean_dictionary = remove_low_means(S[l], empirical_mean_dictionary, K)
        op_count += 1

        S.append(list(empirical_mean_dictionary.keys()))
        epsil_array.append(epsil_array[l] * 3/4)
        delta_array.append(delta_array[l] / 2)
        l += 1
        old_t = new_t

    return S[-1], op_count


def test_bounded_ME_with_Tracking():
    # Test case
    np.random.seed(42)
    A_arms = np.random.normal(size=(1000, 1000))
    query_vector = np.random.normal(size=1000)
    epsilon = 0.02 # suboptimality
    delta = 0.1 # probability margin # TODO seems to have no effect
    K = 1
    
    S, count = BoundedMEwithTracking(epsilon, delta, A_arms, query_vector, K)
    
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
    print("vs " + str(A_arms.shape[0] * A_arms.shape[1]) + "... "\
          + str(count / (A_arms.shape[0] * A_arms.shape[1])))
    
test_bounded_ME_with_Tracking()

