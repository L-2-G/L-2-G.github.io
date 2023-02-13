# import internal Python modules
import sys

# import external modules
#import cmocean
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from gpu_ising import *
from gym import spaces
#import pickle
#import copy

#Supress warning for gpu utilization on small Ising system sizes (they don't use much gpu at all)
import warnings
from numba.cuda.dispatcher import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def to_rgb(x):
  """Converts a 4-channel rgba image into a 3-channel rgb image"""
  # assume rgb premultiplied by alpha
  rgb, a = x[..., 1:4], x[...,0:1]
  return 255-a+rgb

def fig2img(fig,axs,crop=False,fixrgb=True):
    """This converts a matplotlib figure into an image, note: it runds kind of slowly
    Paramaters:
    fig -- the figure you get from calling fig = plt.figure()
    axs -- the axis you get from calling   axs = plt.gca()
    crop (bool) -- whether or not to crop the output image to remove the tranparent padding
    fixrgb (bool) -- True to call to_rgb on the output rgba image, false to just take the rgb channels
    """
    #ax.axis('off')
    fig.tight_layout()
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    fig.canvas.draw()
    
    argb = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
    argb = argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()
    #if crop:
    #    (_,_,x1,y1,x2,y2) = binbounds([argb[:,:,0]>0])[0][0][0]
    #    return to_rgb(argb[x1:x2,y1:y2]) if fixrgb else argb[x1:x2,y1:y2,1:]
    return to_rgb(argb) if fixrgb else argb[:,:,1:]





class IsingEnv(gym.Env):
    def __init__(self,perobs=5,length=100,seed=1,n=32,N=512):
        """ Paramaters do as follows:
            -perobs sets the number of Lattice updates (per cell) between observations of the agent
            -Length gives the number of observations before the episode ends.
            -seed gives the seed for the ising model simulation.
            -n gives the observation image size (1 value assumed a square).
            -N gives the simulation lattice size.
"""
        #storing the game settings
        self.n,self.N,self.length,self.perobs=n,N,length,perobs
        
        #action space setup
        self.JBmul= [1,1/1.15,1/1.05,1.05,1.15,   1,    1,   1,  1]
        self.mewadd=[0,     0,     0,   0,   0,-0.1,-0.02,0.02,0.1]
        
        #ising model sim setup
        self.grid=cuda.to_device(np.array(np.random.random([N,N])>0.5,dtype=np.int32))
        self.act=cuda.to_device(np.zeros_like(self.grid,dtype=np.int32))
        self.outstate=cuda.to_device(np.zeros([3,n,n]).astype('uint8'))
        self.rngs=create_xoroshiro128p_states(self.grid.size, seed=seed)
        self.store=cuda.to_device(np.zeros(8,dtype=np.float32))
        #assumed 16x16 threads per block
        #blocks1 is for the sim
        self.blocks1=(min(N//32,8), min(N//32,8))
        #blocks2 is for the observation image
        self.blocks2=(n//16, n//16)
        print(self.blocks2,self.blocks1)
        #Set the action and observation space
        self.observation_space = spaces.Box(low=0,high=255,shape=[3,n,n],dtype='uint8')
        self.action_space= spaces.Discrete(9)
        #distance is used in reward calculation
        self.distance=25
        self.reset()
        
    def step(self,action):
        """Lets the agent increase/decrease the temperature xor external field and performs some
        (~2x perobs per cell) monte carlo grid updates. Reward is the difference in distance to.
        The specified point in E-M phase space.

        

        '''
        Update the environment by performing an action.

        Inputs:
            action (int) - index of the action to perform.
                Action:         [0,     1,      2,      3,      4,      5,      6,      7,      8]
                JB multiplier:  [1,     1/1.15, 1/1.05, 1.05,   1.15,   1,      1,      1,      1]
                Mu addition:    [0,     0,      0,      0,      0,      -0.1,   -0.02,  0.02,   0.1]
        
        Outputs:
            state - shape [3,n,n] numpy array
                A numpy array containing state variables.
            reward (float) - reward given during this timestep
                The amount of reward generated in perfoming the action.
            done (bool) - Whether or not the final state has been reached and the episode is completed
            params (dict) - unused

        '''
        """
        self.idx+=1
        #action gives change to inverse temp and mew
        self.JB*=self.JBmul[action]
        self.mew+=self.mewadd[action]
        #keep temperature and field strength within reasonable limits
        self.JB=max(min(self.JB,5.0),0.05)
        self.mew=max(min(self.mew,4.0),-4.0)
        
        #update the grid with our temperature and field strength (note this will not reach equillibrium generally)
        continuous_ising[self.blocks1,(16,16)](
            self.grid,
            self.act,
            self.perobs*10,
            self.JB,
            self.mew,
            self.rngs,
            self.N
        )
        #create the observation image using the grid & target E-M point
        zero[1,1](self.store)
        E_S[(self.N//16,self.N//16),(16,16)](self.grid,self.store,self.N)
        set_store[1,1](self.store,self.idx*self.perobs/2,self.JB,self.mew,self.E,self.M,self.N)
        toimg[self.blocks2,(16,16)](self.grid,self.outstate,self.store,self.N,self.n)
        #TL->time TR->M BL-> T,BR->mew for channel 2
        #Target E on left, M on right for channel 3
        outstate = self.outstate.copy_to_host()
        #recalculate the L2 distance in E-M phase space in order to generate the reward
        oldist=self.distance
        self.distance=(
                (self.E-self.store[0]/self.N**2)**2+
                (self.M-(2*self.store[1]/self.N**2-1))**2
            )**0.5*100
            
        return outstate, oldist-self.distance, self.idx>=self.length, {}
          
    def reset(self):
        '''
        Initialize the environment. This sets the ising model back to the phase transition temperature and 
        lets it reach ~equilibrium, then chooses a random magnetization and energy to target.

        Outputs:
            state - shape [3,n,n] numpy array
        '''
        #reset the grid to the critical temperature and no external field
        continuous_ising[self.blocks1,(16,16)](self.grid,self.act,200,0.01,0,self.rngs,self.N)
        continuous_ising[self.blocks1,(16,16)](self.grid,self.act,500,0.4407,0,self.rngs,self.N)
        self.JB=0.4407
        self.mew=0
        #self.rew actually just tells you the return of the last episode and is used for debugging
        self.rew=self.distance
        #reset the distance from the target point
        self.distance=25
        #generate a random accessable point in phase space
        self.M=np.random.random()*2-1
        self.E=-np.random.random()*2
        while self.E>-2*self.M**2:
            self.M=np.random.random()*2-1
            self.E=-np.random.random()*2
        
        self.idx=0
        E_S[self.blocks1,(16,16)](self.grid,self.store,self.N)
        set_store[1,1](self.store,self.idx*self.perobs/2,self.JB,self.mew,self.E,self.M,self.N)
        toimg[self.blocks2,(16,16)](self.grid,self.outstate,self.store,self.N,self.n)
        
        outstate = self.outstate.copy_to_host()
        
        return outstate
    
    def _prerender(self):
        if False:
            fig,ax = plt.subplots(1)
            xs=np.linspace(-1,1,20)
            ys=-xs**2*2
            plt.plot(xs,ys,'r',label="$y=-2x^2$")
            #plt.plot(np.array(M)/128**2*2-1,np.array(E)/128**2,'b--')
            plt.plot(self.M,self.E,'r.',ms=20)
            ME=self.store.copy_to_host()
            plt.plot((2*ME[1]/self.N**2-1),ME[0]/self.N**2,"b.",ms=15)
            plt.title("E-M Phase Space Position")
            plt.ylabel("Energy per site")
            plt.xlabel("Magnetization per site")
            
        ME=self.store.copy_to_host()
        
        arr=np.ones([400,400,3])*255
        if self.N<=400:
            arr[:self.N,:self.N,:]=self.grid.copy_to_host()[:,:,None]*128+127
        else:
            arr[:,:,:]=self.grid.copy_to_host()[:400,:400,None]*128+127
        
        X,Y=np.meshgrid(np.arange(400),np.arange(400))
        
        px = ME[1]/self.N**2
        py = 1+ME[0]/(2*self.N**2)
        
        arr[:,:,0]*=(1-((X-px*400)**2+(Y-py*400)**2<100))
        
        arr[:,:,1]*=(1-((X-(self.M+1)*200)**2+(Y-(self.E/2+1)*400)**2<100))
        
        return arr
    
    def render(self):
        arr = self._prerender()
        plt.imshow(arr)
        plt.show()
