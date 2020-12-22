#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:02:28 2019

@author: edocchipi97
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:36:37 2019

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
            current_player_card_val = np.random.choice(10) + 1
            current_player_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])
            
            self.player_sum += (current_player_card_val * current_player_card_col)
                
            self.player_goes_bust = self.check_go_bust(self.player_sum)
            
            if self.player_goes_bust:
                r = -1
                self.terminal = True
        else:
            while not self.terminal:
                if self.dealer_sum < 17:
                    current_dealer_card_val = np.random.choice(10) + 1
                    current_dealer_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])
                    self.dealer_sum += (current_dealer_card_val * current_dealer_card_col)
                    self.dealer_goes_bust = self.check_go_bust(self.dealer_sum)
                    
                if self.dealer_goes_bust:
                    r = 1
                    self.terminal = True
                elif self.dealer_sum >= 17:
                    r = self.score_highest_sum()
                elif self.t >= self.max_length:
                    r = self.score_highest_sum()
        
        self.t += 1
        self.ret += r
        
        return self.get_state(), r, self.terminal
    
    def check_go_bust(self, card_sum):
        return ((card_sum > 21) or (card_sum < 1))
    
    def score_highest_sum(self):
        r = 0
        if self.dealer_sum > self.player_sum:
            r = -1
        elif self.dealer_sum < self.player_sum:
            r = 1
        else:
            self.terminal = True
        return r

def moving_average(data, window_size):
    data_avg = []
    avg_mask = np.ones(window_size)/window_size
    data_avg = np.convolve(data, avg_mask, 'valid')
    
    return data_avg

## Monte Carlo -- one episode
def Monte_Carlo(Q, Returns, count_state, count_state_action):
#    appeared = np.zeros([10, 21, 2], dtype=int)
    s = env.reset()
    actions = []
    rewards = [] 
    states = [s]
    
    while True:
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1 #s[0] = value of dealers card s[1] = value of players card
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, s[1]-1])
        action = np.random.choice([action_greedy, 1 - action_greedy], p=[1. - epsilon/2., epsilon/2.])
        count_state_action[s[0]-1, s[1]-1, action] += 1
        
        s, r, term = env.step(action, s)
        
        actions.append(action)
        rewards.append(r)
        
        if term: 
            break
        else: 
            states.append(s)
    
    final_reward = rewards[-1]
    sum_reward = np.sum(rewards)

    for t, s in enumerate(states):
        
        dealer_sum = s[0]
        player_sum = s[1]
        dealer_idx = dealer_sum - 1
        player_idx = player_sum - 1
        alpha = 1.0 /count_state_action[dealer_idx, player_idx, actions[t]] # count state action is 0.something and so have to add 1
        
        Returns[dealer_idx, player_idx, actions[t]] += rewards[t]
        Q[dealer_idx, player_idx, actions[t]] += alpha * (rewards[t] - Q[dealer_idx, player_idx, actions[t]])

    return Q, Returns, count_state, count_state_action, sum_reward, final_reward




## Monte Carlo
QMC = np.zeros([10, 21, 2]) # dealer initial card, current player sum, action : Q(s, a)
Returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state_action_mean = np.zeros([10, 21, 2], dtype=int)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_state_mean = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

num_episodes = 100000
env = Easy21()
repeated_runs = 90

cumulative_rewards_MC = np.zeros([repeated_runs, num_episodes])
rewards = np.zeros([repeated_runs, num_episodes])

for instances in range(repeated_runs):
    
    print(instances)
    QMC = np.zeros([10, 21, 2])
    count_state = np.zeros([10, 21], dtype=int) # N(s)
    count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
    
    for i_epi in range(num_episodes):
        QMC, Returns, count_state, count_state_action, reward_MC, last_R = Monte_Carlo(QMC, Returns, count_state, count_state_action)   
        
        rewards[instances, i_epi] = last_R
        
        # plot cumulative reward vs number of episodes
        if i_epi == 0:
            cumulative_rewards_MC[instances, i_epi] = reward_MC
        else:
            cumulative_rewards_MC[instances, i_epi] = cumulative_rewards_MC[instances, i_epi-1] + reward_MC
#        
VMC = QMC.max(axis=2)


# Monte Carlo -- plot
s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, VMC, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card")
ax.set_ylabel("player's sum")
ax.set_zlabel("state value")
plt.title('Monte-Carlo, Optimal Value function')
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()
plt.show()

average_rewards = np.mean(rewards, axis = 0)
std_reward = np.std(rewards, axis = 0)
cum_rewards = np.mean(cumulative_rewards_MC, axis=0)


smoothed_average = moving_average(average_rewards, 1000)
smoothed_std = moving_average(std_reward, 1000)
smoothed_cum = moving_average(cum_rewards, 1000)


figure3 = plt.figure(figsize = (14,6))
plt.subplot(121)
plt.plot(smoothed_average, color = 'deepskyblue')
plt.xlabel('Number of episodes', size = 14)
plt.ylabel('Mean', size = 14)
plt.title('Mean rewards, MC, 100000 episodes', size = 14)
plt.legend()
#plt.savefig('MC_mean_reward.png')


#figure4 = plt.figure(figsize = (10,6))
plt.subplot(122)
plt.plot(smoothed_std, color = 'deepskyblue')
plt.xlabel('Number of episodes', size = 14)
plt.ylabel('Standard deviation', size = 14)
plt.legend()
plt.title('Std, MC, 100000 episodes', size = 14)
plt.savefig('MC_mean_std_reward_agne.png')


# plot reward vs number of episodes
figure5 = plt.figure(figsize=(10, 6))
plt.plot(smoothed_cum)
