import numpy as np
import math

def compute_m(delta, epsilon, A_arms):
    b = np.max(A_arms, axis=0)
    a = np.min(A_arms, axis=0)
    
    u = ((np.log(1/delta) / 2) * ((b - a) ** 2) / (epsilon ** 2))[0]
    m = min((u + 1) / (1 + len(A_arms) * u), (u + u) / (1 + len(A_arms) * u))
    return m

def pull_number_modifier(current_epsilon, current_delta, S_size, K):
    epsilon_coefficient = 2 / (current_epsilon ** 2)
    log_term_numerator = 2 * (S_size - K)
    log_term_denominator = current_delta * ((S_size - K) / 2 + 1)
    return epsilon_coefficient * np.log(log_term_numerator / log_term_denominator)

def remove_low_means(S, mean_dict, K):
    threshold_margin = math.ceil((len(S) - K) / 2)
    sorted_mean_dict = sorted(mean_dict.items(), key=lambda x: x[1], reverse=True)
    
    threshold_mean = sorted_mean_dict[threshold_margin - 1][1]
    new_mean_dict = dict(filter(lambda x: x[1] >= threshold_mean, mean_dict.items()))
    return new_mean_dict

# Bounded Median Estimator
def BoundedME(epsilon, delta, A_arms, query_vector, K=1):
    # For tracking floating point operations count
    op_count = 0

    S_current = A_arms
    current_epsilon = epsilon / 4
    current_delta = delta / 2
    reward_dictionary = {tuple(a): 0 for a in A_arms}
    
    m = compute_m(delta, epsilon, A_arms)
    old_t = 0
    while len(S_current) > K:
        new_t = math.floor(m * pull_number_modifier(current_epsilon, current_delta, len(S_current), K))
        op_count += 6

        reward_dictionary, ops = pull_new_arms(query_vector, K, S_current, reward_dictionary, old_t, new_t)
        op_count += ops

        S_current = list(reward_dictionary.keys())
        current_epsilon = current_epsilon * 3/4
        current_delta = current_delta / 2
        old_t = new_t
        op_count += 3

    return S_current, op_count

def pull_new_arms(query_vector, K, S_current, reward_dictionary, old_t, new_t):
    ops = 0
    for a in S_current:
        arm_pulls = np.dot(query_vector[old_t:new_t], a[old_t:new_t])
        reward = arm_pulls # / (new_t - old_t - 1) # no need for actual mean
        ops += query_vector[old_t:new_t].size  # Counting multiplications inside np.dot

        reward_dictionary[tuple(a)] += reward

    reward_dictionary = remove_low_means(S_current, reward_dictionary, K)
    ops += 2
    return reward_dictionary, ops


