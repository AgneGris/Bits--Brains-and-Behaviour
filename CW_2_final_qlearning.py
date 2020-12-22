#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:53:44 2019

@author: edocchipi97
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Easy21:

    def __init__(self, verbose=False, max_length=1000):
        self.max_length = max_length
        self.verbose = verbose
        
    def reset(self):
        player_first_card_val = np.random.choice(10) + 1
        dealer_first_card_val = np.random.choice(10) + 1
        
        self.player_sum = player_first_card_val
        self.dealer_sum = dealer_first_card_val
        
        self.player_goes_bust = False
        self.dealer_goes_bust = False

        self.ret = 0
        self.terminal = False
        self.t = 0
        
        if self.verbose:
            print("Initial state: ", self.get_state())
        return self.get_state()
    
    
    def get_state(self):
        return [self.dealer_sum, self.player_sum]

    
    def step(self, action, state):
        # action 1: hit   0: stick
        # color: 1: black   -1: red
        r = 0

        if action:
            if self.verbose:
                print("Player action")
            current_player_card_val = np.random.choice(10) + 1
            current_player_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])
            
            self.player_sum += (current_player_card_val * current_player_card_col)
            
            if self.verbose:
                print("player drew: ", current_player_card_val * current_player_card_col, self.player_sum)
                
            self.player_goes_bust = self.check_go_bust(self.player_sum)
            
            if self.player_goes_bust:
                if self.verbose:
                    print("player bust")
                r = -1
                self.terminal = True
        else:
            if self.verbose:
                print("dealer action")
            while not self.terminal:
                if self.dealer_sum < 17:
                    current_dealer_card_val = np.random.choice(10) + 1
                    current_dealer_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])
                    self.dealer_sum += (current_dealer_card_val * current_dealer_card_col)
                    if self.verbose:
                        print("dealer drew: ", current_dealer_card_val * current_dealer_card_col, self.dealer_sum)
                    self.dealer_goes_bust = self.check_go_bust(self.dealer_sum)
                    
                if self.dealer_goes_bust:
                    if self.verbose:
                        print("dealer bust")
                    r = 1
                    self.terminal = True
                elif self.dealer_sum >= 17:
                    if self.verbose:
                        print("dealer done")
                    r = self.score_highest_sum()
                elif self.t >= self.max_length:
                    if self.verbose:
                        print("too many steps")
                    r = self.score_highest_sum()
        
        if self.verbose:
            print("dealing done")
        self.t += 1
        self.ret += r
        
        if self.verbose:
            print(f"state: {self.get_state()}, reward: {r}, term: {self.terminal}, act: {action}")
            print("===============================================")
        return self.get_state(), r, self.terminal
    
    def check_go_bust(self, card_sum):
        return ((card_sum > 21) or (card_sum < 1))
    
    def score_highest_sum(self):
        r = 0
        if self.dealer_sum > self.player_sum:
            if self.verbose:
                print("dealer wins")
            r = -1
        elif self.dealer_sum < self.player_sum:
            if self.verbose:
                print("player wins")
            r = 1
        else:
            if self.verbose:
                print("draw")
        self.terminal = True
        return r


def Q_learning(Q, count_state, count_state_action):
    
    s = env.reset()
    ret = 0
    rewards = []
    
    while True:
        
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1 
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, s[1]-1])
        action = np.random.choice([action_greedy, 1 - action_greedy], p=[1. - epsilon/2., epsilon/2.])
        count_state_action[s[0]-1, s[1]-1, action] += 1
        
        alpha = 1/count_state_action[s[0]-1, s[1]-1, action]
        
        s_prime, r, term = env.step(action, s)
        rewards.append(r)
        
        ret = rewards[-1]
        
        if not term:
            action_greedy_prime = Q[s_prime[0]-1, s_prime[1]-1, :].argmax()
            
            Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + Q[s_prime[0]-1, s_prime[1]-1, action_greedy_prime] -  Q[s[0]-1,s[1]-1, action])
            
            s = s_prime
        
        if term:
            
             Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + 0 - Q[s[0]-1,s[1]-1, action])
             break
         
    return Q, count_state, count_state_action, ret

def Q_learning_epsilon(Q, count_state, count_state_action, epsilon):
    
    s = env.reset()
    ret = 0
    rewards = []
    
    while True:
        
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1 
        action = np.random.choice([action_greedy, 1 - action_greedy], p=[1. - epsilon/2., epsilon/2.])
        count_state_action[s[0]-1, s[1]-1, action] += 1
        
        alpha = 1/count_state_action[s[0]-1, s[1]-1, action]
            
        s_prime, r, term = env.step(action, s)
        rewards.append(r)
        
        ret = rewards[-1]
        
        if not term:
            action_greedy_prime = Q[s_prime[0]-1, s_prime[1]-1, :].argmax()
            
            
            Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + Q[s_prime[0]-1, s_prime[1]-1, action_greedy_prime] -  Q[s[0]-1,s[1]-1, action])
            
            s = s_prime
        
        if term:
            
             Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + 0 - Q[s[0]-1,s[1]-1, action])
             break
         
    return Q, count_state, count_state_action, ret

def moving_average(data, window_size):
    data_avg = []
    avg_mask = np.ones(window_size)/window_size
    data_avg = np.convolve(data, avg_mask, 'valid')
    
    return data_avg


## Q learning
Q_l = np.zeros([10, 21, 2]) # dealer initial card, current player sum, action : Q(s, a)
Returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

n_episodes = 80000
env = Easy21()
repeated_runs = 100

cumulative_rewards = np.zeros(n_episodes)
rewards = np.zeros([repeated_runs, n_episodes])

for instances in range(repeated_runs):
    
    print(instances)
    Q_l = np.zeros([10, 21, 2])
    count_state = np.zeros([10, 21], dtype=int) # N(s)
    count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)

    for i_epi in range(n_episodes):
        
        Q_l, count_state, count_state_action, last_R = Q_learning(Q_l, count_state, count_state_action)   
        rewards[instances, i_epi] = last_R
    
        if i_epi == 0:
            cumulative_rewards[i_epi] = last_R
        
        else:
            cumulative_rewards[i_epi] = cumulative_rewards[i_epi-1] + last_R
        
V_l = Q_l.max(axis=2)

epsilon_list = [0.05, 0.25, 0.75]

rewards_epsilon = np.zeros([repeated_runs, n_episodes, len(epsilon_list)])
cumulative_rewards_epsilon = np.zeros([repeated_runs, n_episodes, len(epsilon_list)])

for epsilon in range(len(epsilon_list)):
    
    for instances in range(repeated_runs):
        
        print(instances)
        Q_l = np.zeros([10, 21, 2])
        count_state = np.zeros([10, 21], dtype=int) # N(s)
        count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
    
        for i_epi in range(n_episodes):
            
            Q_l, count_state, count_state_action, last_R = Q_learning_epsilon(Q_l, count_state, count_state_action, epsilon_list[epsilon])   
            rewards_epsilon[instances, i_epi, epsilon] = last_R
            
            if i_epi == 0:
                cumulative_rewards_epsilon[instances, i_epi, epsilon] = last_R
        
            else:
                cumulative_rewards_epsilon[instances, i_epi, epsilon] = cumulative_rewards_epsilon[instances, i_epi-1, epsilon] + last_R
        
# Q_learning -- plot
s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, V_l, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card")
ax.set_ylabel("player's sum")
ax.set_zlabel("state value")
plt.title('Value function, Q-Learning')
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()


plt.show()


average_rewards_epsilon1 = np.mean(rewards_epsilon[:,:,0], axis = 0)
average_rewards_epsilon2 = np.mean(rewards_epsilon[:,:,1], axis = 0)
average_rewards_epsilon3 = np.mean(rewards_epsilon[:,:,2], axis = 0)

average_rewards = np.mean(rewards, axis = 0)

std_rewards_epsilon1 = np.std(rewards_epsilon[:,:,0], axis = 0)
std_rewards_epsilon2 = np.std(rewards_epsilon[:,:,1], axis = 0)
std_rewards_epsilon3 = np.std(rewards_epsilon[:,:,2], axis = 0)

std_rewards = np.std(rewards, axis = 0)


smoothed_average_epsilon1 = moving_average(average_rewards_epsilon1, 1000)
smoothed_average_epsilon2 = moving_average(average_rewards_epsilon2, 1000)
smoothed_average_epsilon3 = moving_average(average_rewards_epsilon3, 1000)
smoothed_average = moving_average(average_rewards, 1000)

smoothed_std_epsilon1 = moving_average(std_rewards_epsilon1, 10000)
smoothed_std_epsilon2 = moving_average(std_rewards_epsilon2, 10000)
smoothed_std_epsilon3 = moving_average(std_rewards_epsilon3, 10000)
smoothed_std = moving_average(std_rewards, 10000)


figure1 = plt.figure(figsize = (10,6))
plt.plot(smoothed_average, label = 'changing')
plt.plot(smoothed_average_epsilon1, label = 'epsilon = 0.05')
plt.plot(smoothed_average_epsilon2, label = 'epsilon = 0.25')
plt.plot(smoothed_average_epsilon3, label = 'epsilon = 0.75')
plt.legend()

plt.xlabel('# Episodes', size = 14)
plt.ylabel('Mean', size = 14)
plt.title('Mean rewards - Q_learning, 80000 episodes, 100 repeated runs', size = 14)
plt.savefig('Q_learning_mean_reward.png')

figure2 = plt.figure(figsize = (10,6))
plt.plot(smoothed_std, label = 'changing')
plt.plot(smoothed_std_epsilon1, label = 'epsilon = 0.05')
plt.plot(smoothed_std_epsilon2, label = 'epsilon = 0.25')
plt.plot(smoothed_std_epsilon3, label = 'epsilon = 0.75')
plt.legend()

plt.xlabel('Number of episodes', size = 14)
plt.ylabel('Standard deviation', size = 14)
plt.title('Standard deviation of rewards - Q_learning, 80000 episodes, 100 repeated runs', size = 14)
plt.savefig('Q_learning_std_reward.png')


#cumulative_rewards_epsilon1 = np.mean(cumulative_rewards_epsilon[:,:,0], axis = 0)
#cumulative_rewards_epsilon2 = np.mean(cumulative_rewards_epsilon[:,:,1], axis = 0)
#cumulative_rewards_epsilon3 = np.mean(cumulative_rewards_epsilon[:,:,2], axis = 0)
#cumulative_rewards_mean = np.mean(cumulative_rewards, axis = 0)
#
#smoothed_cum_epsilon1 = moving_average(cumulative_rewards_epsilon1, 1000)
#smoothed_cum_epsilon2 = moving_average(cumulative_rewards_epsilon2, 1000)
#smoothed_cum_epsilon3 = moving_average(cumulative_rewards_epsilon3, 1000)
#smoothed_cum = moving_average(cumulative_rewards_mean, 1000)
#
#
#figure3 = plt.figure(figsize = (10,6))
#plt.plot(smoothed_cum, label = 'changing')
#plt.plot(smoothed_cum_epsilon1, label = 'epsilon = 0.05')
#plt.plot(smoothed_cum_epsilon2, label = 'epsilon = 0.25')
#plt.plot(smoothed_cum_epsilon3, label = 'epsilon = 0.75')
#
#plt.xlabel('Number of episodes', size = 14)
#plt.ylabel('Standard deviation', size = 14)
#plt.title('Cumulative rewards - Q_learning, 80000 episodes, 100 repeated runs', size = 14)
#plt.savefig('Q_learning_cumulative_reward.png')