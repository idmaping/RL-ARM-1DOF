import cv2
import numpy as np
from numpy import linalg as LA

class ENVIRONMENT:
    def __init__(self):
        self.ENV_WIDTH = 100
        self.ENV_HEIGHT = 100
        self.ENV_BACKGROUND = np.zeros((self.ENV_WIDTH,self.ENV_HEIGHT,3), np.uint8)
        
        self.TGT_X = 0
        self.TGT_Y = 0
        self.TGT_THETA = 180
        self.TGT_RHO = 40
        

        self.POS_THETA = 180
        self.POS_RHO = 40
        self.POS_X = (self.ENV_WIDTH/2) + self.POS_RHO * np.cos(np.radians(self.POS_THETA))
        self.POS_Y = (self.ENV_WIDTH/2) + self.POS_RHO * -np.sin(np.radians(self.POS_THETA))
        
        
        self.DIM_ACTION = 2
        self.DIM_STATE = 1
        
        self.RL_STATE = [0]
        self.RL_REWARD = 0

    def getReward(self):
        self.POS_X = (self.ENV_WIDTH/2) + self.POS_RHO * np.cos(np.radians(self.POS_THETA))
        self.POS_Y = (self.ENV_WIDTH/2) + self.POS_RHO * -np.sin(np.radians(self.POS_THETA))
        reward = -LA.norm([[self.POS_X,self.POS_Y],[self.TGT_X,self.TGT_X]])
        return reward

    def step(self,action):
        self._POS_RHO = self.POS_RHO
        if action == 0:
            self.POS_THETA+=10

        elif action == 1:
            self.POS_THETA-=10
    
    def show(self):
        temp = self.ENV_BACKGROUND.copy()
        temp = cv2.line(temp, (50,50), (int(self.POS_X),int(self.POS_Y)), (0,255,0), 1)
        temp = cv2.circle(temp, (int(self.TGT_X),int(self.TGT_Y)), 2, (255,255,255), 2)
        temp = cv2.resize(temp, ((200,200)), interpolation = cv2.INTER_AREA)
        
        cv2.imshow('ENVIRONMENT',temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    env = ENVIRONMENT()
    env.step(1)
    print("st1",env.getReward())
    env.show()

    env.step(1)
    print("st2",env.getReward())
    env.show()
    
    env.step(1)
    print("st3",env.getReward())
    env.show()

    env.step(1)
    print("st4",env.getReward())
    env.show()

    env.step(1)
    print("st5",env.getReward())
    env.show()

    env.step(1)
    print("st6",env.getReward())
    env.show()

    env.step(1)
    print("st7",env.getReward())
    env.show()

    env.step(1)
    print("st8",env.getReward())
    env.show()

    env.step(1)
    print("st9",env.getReward())
    env.show()
    
