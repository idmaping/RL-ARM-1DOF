import cv2
import numpy as np
from numpy import linalg as LA

class ENVIRONMENT:
    def __init__(self):
        self.ENV_WIDTH = 20
        self.ENV_HEIGHT = 20
        self.ENV_BACKGROUND = np.zeros((self.ENV_WIDTH,self.ENV_HEIGHT,3), np.uint8)
        
        self.TGT_X = 0
        self.TGT_Y = 0
        self.TGT_THETA = np.arctan2(50-self.TGT_Y,self.TGT_X - 50) * 180 / np.pi 
        self.TGT_RHO = 5 #gaperlu
        
        self.POS_THETA = 180
        self.POS_RHO = 5
        self.POS_X = (self.ENV_WIDTH/2) + self.POS_RHO * np.cos(np.radians(self.POS_THETA))
        self.POS_Y = (self.ENV_WIDTH/2) + self.POS_RHO * -np.sin(np.radians(self.POS_THETA))
        
        self.DIM_ACTION = 2
        self.DIM_STATE = 3
        
        self.RL_STATE = [0]
        self.RL_REWARD = 0

        self.REWARD_GOAL = 700

    def getReward(self):
        
        ## REWARD SYSTEM V1
        #self.POS_X = (self.ENV_WIDTH/2) + self.POS_RHO * np.cos(np.radians(self.POS_THETA))
        #self.POS_Y = (self.ENV_WIDTH/2) + self.POS_RHO * -np.sin(np.radians(self.POS_THETA))
        #reward = (-LA.norm([[self.POS_X,self.POS_Y],[self.TGT_X,self.TGT_X]]))/1 + 35

        ## REWARD SYSTEM V2
        self.TGT_THETA = np.arctan2((self.ENV_HEIGHT/2)-self.TGT_Y,self.TGT_X - (self.ENV_WIDTH/2)) * 180 / np.pi 
        if self.TGT_THETA <0:
            self.TGT_THETA = 360 + self.TGT_THETA
        
        reward = -10
        angle_thresh = 5
        if self.POS_THETA-angle_thresh <= self.TGT_THETA <= self.POS_THETA+angle_thresh:
            reward = self.REWARD_GOAL
        
        return reward

    def getState(self):
        return (self.TGT_X,self.TGT_Y,self.POS_THETA)

    def reset(self):
        self.TGT_X = np.random.randint(self.ENV_WIDTH)
        self.TGT_Y = np.random.randint(self.ENV_HEIGHT)
        self.POS_THETA = np.random.randint(71)*5


    def step(self,action):
        self._POS_RHO = self.POS_RHO
        if action == 0:
            self.POS_THETA+=5

        elif action == 1:
            self.POS_THETA-=5

        if self.POS_THETA>=360:
            self.POS_THETA = self.POS_THETA-360

        if self.POS_THETA<0:
            self.POS_THETA = 360 + self.POS_THETA

        

    def show(self):
        multi = 10
        temp = self.ENV_BACKGROUND.copy()
        
        temp = cv2.resize(temp, ((temp.shape[0]*multi,temp.shape[1]*multi)), interpolation = cv2.INTER_AREA)

        POS_X = (self.ENV_WIDTH/2) + self.POS_RHO * np.cos(np.radians(self.POS_THETA))
        POS_Y = (self.ENV_HEIGHT/2) + self.POS_RHO * -np.sin(np.radians(self.POS_THETA))
        
        temp = cv2.line(temp, (int(self.ENV_WIDTH/2)*multi,int(self.ENV_HEIGHT/2)*multi), (int(POS_X)*multi,int(POS_Y)*multi), (0,255,0), 2)
        temp = cv2.circle(temp, (int(self.TGT_X)*multi,int(self.TGT_Y)*multi), 3, (255,255,255), 3)
        
        temp = cv2.resize(temp, ((500,500)), interpolation = cv2.INTER_AREA)
        
        cv2.imshow('ENVIRONMENT',temp)
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    env = ENVIRONMENT()
    env.step(1)
    print("st1",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st2",env.getReward(),env.getState())
    env.show()
    
    env.step(1)
    print("st3",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st4",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st5",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st6",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st7",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st8",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()

    env.step(1)
    print("st9",env.getReward(),env.getState())
    env.show()
    
