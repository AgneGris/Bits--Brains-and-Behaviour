
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

## Monte Carlo -- one episode
def monte_carlo(Q, Returns, count_state, count_state_action):
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
        
        Returns[dealer_idx, player_idx, actions[t]] += rewards[-1]
        Q[dealer_idx, player_idx, actions[t]] += alpha * (rewards[-1] - Q[dealer_idx, player_idx, actions[t]])

    return Q, Returns, count_state, count_state_action, sum_reward, final_reward

def moving_average(data, window_size):
    data_avg = []
    avg_mask = np.ones(window_size)/window_size
    data_avg = np.convolve(data, avg_mask, 'valid')
    
    return data_avg


def sarsa(Q, count_state, count_state_action):
    
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
            
            action2_greedy = Q[s_prime[0]-1, s_prime[1]-1, :].argmax()
            epsilon = count_constant / float(count_constant + count_state[s[0]-1, s[1]-1])
            action2 = np.random.choice([action2_greedy, 1 - action2_greedy], p=[1. - epsilon/2., epsilon/2.])
            
            count_state[s_prime[0]-1, s_prime[1]-1] += 1
            count_state_action[s_prime[0]-1, s_prime[1]-1, action2] += 1
            

            Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + Q[s_prime[0]-1, s_prime[1]-1, action2] -  Q[s[0]-1,s[1]-1, action])
            
            s = s_prime
        
        if term:
            
             Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + 0 - Q[s[0]-1,s[1]-1, action])
             break
         
    return Q, count_state, count_state_action, ret


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
            count_state[s_prime[0]-1, s_prime[1]-1] += 1
            action_greedy_prime = Q[s_prime[0]-1, s_prime[1]-1, :].argmax()
            count_state_action[s_prime[0]-1, s_prime[1]-1, action_greedy_prime] += 1
            
            
            Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + Q[s_prime[0]-1, s_prime[1]-1, action_greedy_prime] -  Q[s[0]-1,s[1]-1, action])
            
            s = s_prime
        
        if term:
            
             Q[s[0]-1,s[1]-1, action] =  Q[s[0]-1,s[1]-1, action] + alpha*(r + 0 - Q[s[0]-1,s[1]-1, action])
             break
         
    return Q, count_state, count_state_action, ret


# Monte Carlo -- plot


## Monte Carlo
Q_MC = np.zeros([10, 21, 2]) # dealer initial card, current player sum, action : Q(s, a)
Returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state_action_mean = np.zeros([10, 21, 2], dtype=int)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_state_mean = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

n_episodes = 200000
env = Easy21()
repeated_runs = 90

cumulative_rewards_MC = np.zeros([repeated_runs, n_episodes])
rewards_MC = np.zeros([repeated_runs, n_episodes])

V_MC = Q_MC.max(axis=2)
s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')


average_rewards_MC = np.mean(rewards_MC, axis = 0)
std_reward_MC = np.std(rewards_MC, axis = 0)

smoothed_average_MC = moving_average(average_rewards_MC, 1000)
smoothed_std_MC = moving_average(std_reward_MC, 1000)



## Sarsa
Q_sarsa = np.zeros([10, 21, 2]) # dealer initial card, current player sum, action : Q(s, a)
Returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)

count_constant = 100

n_episodes = 200000
env = Easy21()
repeated_runs = 90

cumulative_rewards = np.zeros([repeated_runs, n_episodes])
rewards_sarsa = np.zeros([repeated_runs, n_episodes])

env = Easy21()

for instances in range(repeated_runs):
    
    print(instances)
    Q_sarsa = np.zeros([10, 21, 2])
    count_state = np.zeros([10, 21], dtype=int) # N(s)
    count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)

    for i_epi in range(n_episodes):
        
        Q_sarsa, count_state, count_state_action, last_R = sarsa(Q_sarsa, count_state, count_state_action)   
        rewards_sarsa[instances, i_epi] = last_R
    
        if i_epi == 0:
            cumulative_rewards[instances, i_epi] = last_R
        
        else:
            cumulative_rewards[instances, i_epi] = cumulative_rewards[instances, i_epi-1] + last_R
        
V_sarsa = Q_sarsa.max(axis=2)

average_rewards_sarsa = np.mean(rewards_sarsa, axis = 0)
std_rewards_sarsa = np.std(rewards_sarsa, axis = 0)

smoothed_average_sarsa = moving_average(average_rewards_sarsa, 10000)
smoothed_std_sarsa = moving_average(std_rewards_sarsa, 10000)

## Q learning
Q_l = np.zeros([10, 21, 2]) # dealer initial card, current player sum, action : Q(s, a)
Returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

n_episodes = 20000
env = Easy21()
repeated_runs = 90

cumulative_rewards = np.zeros(n_episodes)
rewards_ql = np.zeros([repeated_runs, n_episodes])

for instances in range(repeated_runs):
    
    print(instances)
    Q_l = np.zeros([10, 21, 2])
    count_state = np.zeros([10, 21], dtype=int) # N(s)
    count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)

    for i_epi in range(n_episodes):
        
        Q_l, count_state, count_state_action, last_R = Q_learning(Q_l, count_state, count_state_action)   
        rewards_ql[instances, i_epi] = last_R
    
        if i_epi == 0:
            cumulative_rewards[i_epi] = last_R
        
        else:
            cumulative_rewards[i_epi] = cumulative_rewards[i_epi-1] + last_R
        
V_l = Q_l.max(axis=2)


average_rewards_ql = np.mean(rewards_ql, axis = 0)
std_rewards_ql = np.std(rewards_ql, axis = 0)

smoothed_std_ql = moving_average(std_rewards_ql, 10000)
smoothed_average_ql = moving_average(average_rewards_ql, 10000)



# all three means

figure1 = plt.figure(figsize=(10,6))

plt.plot(smoothed_average_MC,label = 'MC')
plt.plot(smoothed_average_sarsa, label = 'SARSA')
plt.plot(smoothed_average_ql, label = 'Q-L')
plt.legend()
plt.savefig('mean_3.png')


# all three variances

figure2 = plt.figure(figsize=(10,6))

plt.plot(smoothed_std_MC,label = 'MC')
plt.plot(smoothed_std_sarsa, label = 'SARSA')
plt.plot(smoothed_std_ql, label = 'Q-L')
plt.legend()
plt.savefig('std_3.png')




