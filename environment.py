import cv2
import numpy as np

class ENVIRONMENT:
    def __init__(self):
        self.ENV_WIDTH = 100
        self.ENV_HEIGHT = 100
        self.ENV_BACKGROUND = np.zeros((self.ENV_WIDTH,self.ENV_HEIGHT,3), np.uint8)
        
        self.TGT_X = 0
        self.TGT_Y = 0
        self.TGT_THETA = 180
        self.TGT_RHO = 0
        
        self.POS_X = 0
        self.POS_Y = 0
        self.POS_THETA = 180
        self.POS_RHO = 30
        
        self.DIM_ACTION = 2
        self.DIM_STATE = 1
        
        self.RL_STATE = [0]
        self.RL_REWARD = 0

    def getReward(self):
        reward = 0

        return reward


    def step(self,action):
        if action == 0:
            self.POS_THETA+=5

        elif action == 1:
            self.POS_THETA-=5
    
    def show(self):
        temp = self.ENV_BACKGROUND.copy()

        x = (self.ENV_WIDTH/2) + self.POS_RHO * np.cos(np.radians(self.POS_THETA))
        y = (self.ENV_WIDTH/2) + self.POS_RHO * -np.sin(np.radians(self.POS_THETA))
        
        temp = cv2.line(temp, (50,50), (int(x),int(y)), (0,255,0), 1)
        temp = cv2.resize(temp, ((200,200)), interpolation = cv2.INTER_AREA)
        
        cv2.imshow('ENVIRONMENT',temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    env = ENVIRONMENT()
    #env.show()
