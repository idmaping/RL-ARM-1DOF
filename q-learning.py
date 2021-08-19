import numpy as np
from numpy import linalg as LA
from environment import ENVIRONMENT
import matplotlib.pyplot as plt
import pickle
import time
import cv2

class QLEARNING:
    def __init__(self):
        self.MAX_EPISODE = 200000
        self.MAX_STEP = 400
        self.SHOW_EVERY_EPISODE = 10000
        self.EPSILON_DECAY = 0.9998
        self.EPSILON = 0.9
        self.LR = 0.1
        self.DISCOUNT = 0.95
        #self.start_q_table = None
        self.env = ENVIRONMENT()
        self.q_table = {}
        self.SHOW = False

    def init(self,start_q_table = None):
        if start_q_table is None:
            print("NO Q-TABLE IS LOADED")
            print("CREATING Q-TABLE")
            for i in range(0,self.env.ENV_WIDTH+1):
                for ii in range(0,self.env.ENV_HEIGHT+1):
                    for iii in range(0,360):
                        self.q_table[(i,ii,iii)] = [np.random.uniform(-5, 0) for i in range(self.env.DIM_ACTION)]
            print(self.q_table[(0,0,0)][0])
            print("DONE CREATING Q-TABLE")
        else:
            with open(start_q_table, "rb") as f:
                self.q_table = pickle.load(f)
    
    def test(self,weight = ''):
        self.init(start_q_table=weight)
        while(1):
            self.env.reset()
            for i in range(100):
                state = self.env.getState()
                action = np.argmax(self.q_table[state])
                self.env.step(action)
                reward = self.env.getReward()
                self.env.show()
                print(state,action)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                if reward == self.env.REWARD_GOAL: 
                    break
                
        
        


    def learn(self):
        self.init()
        #print(self.q_table[self.env.getState()][0])
        episode_rewards = []
        epsilons = []
        for episode in range(self.MAX_EPISODE+1):
            
            episode_reward = 0
            self.env.reset()

            if episode % self.SHOW_EVERY_EPISODE == 0 and episode>0 :
                #self.SHOW = True
                with open(f"qtable-{self.env.ENV_HEIGHT}x{self.env.ENV_WIDTH}ep{episode}.pickle", "wb") as f:
                    pickle.dump(self.q_table, f)
            else:
                self.SHOW = False

            for i in range(self.MAX_STEP):
                state = self.env.getState()
                if np.random.random() > self.EPSILON:
                    action = np.argmax(self.q_table[state])
                else:
                    action = np.random.randint(0, self.env.DIM_ACTION)                        
                self.env.step(action)
                _state = self.env.getState()
                max_future_q = np.max(self.q_table[_state])
                current_q = self.q_table[state][action]
                reward = self.env.getReward()

                if reward == self.env.REWARD_GOAL:
                    new_q = reward
                else:
                    new_q = (1 - self.LR) * current_q + self.LR * (reward + self.DISCOUNT * max_future_q)
                self.q_table[state][action] = new_q

                episode_reward += reward

                ## SHOW ANIMATION IN EVERY X EPISODE MECHANISM
                if self.SHOW:
                    #print("EPISODE : ",episode,"    R : ",episode_reward)
                    self.env.show()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                ## BREAK IF REACH GOAL
                if reward == self.env.REWARD_GOAL: 
                    break
            
            episode_rewards.append(episode_reward)
            epsilons.append(self.EPSILON)
            self.EPSILON *= self.EPSILON_DECAY

            print("EPISODE : ",episode,"    R : ",episode_reward)
        
        moving_avg = np.convolve(episode_rewards, np.ones((self.SHOW_EVERY_EPISODE,))/self.SHOW_EVERY_EPISODE, mode='valid')
        
        
        #plt.plot([i for i in range(len(moving_avg))], moving_avg)
        #plt.ylabel(f"Reward {self.SHOW_EVERY_EPISODE}ma")
        #plt.xlabel("episode #")
        #plt.show()

        f, axes = plt.subplots()
        axes.plot([i for i in range(len(moving_avg))], 
                     moving_avg,
                     color='blue',label='REWARD')
        axes.set_ylabel('REWARD', color='tab:blue')
        ax2 = axes.twinx()      
        ax2.plot([i for i in range(len(epsilons))], 
                     epsilons,
                     color='red',label='EPSILON')
        ax2.set_ylabel('EPSILON', color='tab:red')
        axes.set_xlabel("EPISODE")
        plt.show()


        with open(f"qtable-{self.env.ENV_HEIGHT}x{self.env.ENV_WIDTH}ep{episode}.pickle", "wb") as f:
            pickle.dump(self.q_table, f)



if __name__ == '__main__':
    ql=QLEARNING()
    #ql.learn()
    ql.test(weight='qtable-20x20ep200000.pickle')