import numpy as np
import random

class Arm:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev
    
    def pull(self):
        return np.random.normal(self.mean, self.std_dev)

class EpsilonGreedyBandit:
    def __init__(self, arms, epsilon):
        self.arms = arms
        self.epsilon = epsilon
        self.num_arms = len(arms)
        self.num_pulls = np.zeros(self.num_arms)
        self.sum_rewards = np.zeros(self.num_arms)
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_arms)
        else:
            return np.argmax(self.sum_rewards / self.num_pulls)
    
    def play(self, num_rounds):
        total_reward = 0
        for _ in range(num_rounds):
            chosen_arm = self.select_arm()
            reward = self.arms[chosen_arm].pull()
            self.num_pulls[chosen_arm] += 1
            self.sum_rewards[chosen_arm] += reward
            total_reward += reward
        return total_reward

def ucb_bandit(arms, num_rounds):
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)
    sum_rewards = np.zeros(num_arms)
    total_reward = 0
    
    for arm in range(num_arms):
        reward = arms[arm].pull()
        num_pulls[arm] += 1
        sum_rewards[arm] += reward
        total_reward += reward
    
    for _ in range(num_rounds - num_arms):
        ucb_values = sum_rewards / num_pulls + np.sqrt(2 * np.log(_ + num_arms) / num_pulls)
        chosen_arm = np.argmax(ucb_values)
        reward = arms[chosen_arm].pull()
        num_pulls[chosen_arm] += 1
        sum_rewards[chosen_arm] += reward
        total_reward += reward
    
    return total_reward

# Example usage

def epsilon_vs_ucb():
    m = 10
    arms = []
    for i in range(m):
        arms.append(Arm(random.uniform(1,10), random.uniform(1,3)))

    epsilon_greedy_bandit = EpsilonGreedyBandit(arms, epsilon=0.1)
    ucb_reward = ucb_bandit(arms, num_rounds=1000)
    epsilon_greedy_reward = epsilon_greedy_bandit.play(num_rounds=1000)
    print(f"UCB total reward: {ucb_reward}")
    print(f"Epsilon Greedy total reward: {epsilon_greedy_reward}")
    if ucb_reward > epsilon_greedy_reward:
        return 1 # flag
    else:
        return 0 # flage

def main():
    trials = 100
    ucb_wins = 0
    for i in range(trials):
        ucb_wins += epsilon_vs_ucb()
    print("UCB wins: {} over {} trials".format(ucb_wins, trials))

main()

