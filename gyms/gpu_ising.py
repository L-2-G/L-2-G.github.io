import numpy as np
from numba import jit,cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from matplotlib import pyplot as plt
import math,time
from threading import Thread,Lock


@cuda.jit
def fast_ising(grid,JB,mew,parity,rng_states,size):
    """Updates 1/4 of the grid by stochastically choosing points
    on a checkerboard grid and updating with boltzmann probability"""
    #find grid location
    i,j=cuda.grid(2)
    #get your first random number
    n = xoroshiro128p_uniform_float32(rng_states, i*size+j)
    #parity is 0 or 1, and tells you which half of the checkerboard to update
    #Instead of updating the entire checkerboard,  each position
    #is updated with probability 0.5
    if ((i+j)%2==parity and n < 0.5 and i<size and j<size):
        #sum the values of the nearest neighbors
        sum_=grid[i-1][j]+grid[i+1-size][j]+grid[i][j-1]+grid[i][j+1-size]
        #its grid*2-1 because I am using zero for spin down and 1 for spin up
        delta = 2.0 * (grid[i][j]*2-1)*(2*sum_-4+mew)
        #get our second random number
        n = xoroshiro128p_uniform_float32(rng_states, i*size+j)
        #update the cell according to the boltzmann distribution
        if (delta < 0 or 1-n < math.exp(-JB*delta)):
            grid[i][j]=1-grid[i][j]



@cuda.jit
def continuous_ising(grid,active,steps,JB,mew,rng_states,size):
    """Updates the grid by stochastically choosing points
    and updating with boltzmann probability"""
    
    #this runs async on each thread
    x,y=cuda.grid(2)
    idx=x*size+y
    for x in range(steps):
        #choose a random grid location  
        i=int(math.floor(xoroshiro128p_uniform_float32(rng_states, idx)* size))
        j=int(math.floor(xoroshiro128p_uniform_float32(rng_states, idx)* size))
        if i==size or j==size:continue
        #lock this grid position
        lock1=1-cuda.atomic.max(active[i],j,1)
        #threadfence to make sure other blocks know this is locked
        cuda.threadfence()
        #check to make sure none of the neighbours are locked
        lock2 = int(active[i-1][j]+active[i+1-size][j]+active[i][j-1]+active[i][j+1-size]==0)
        #check the spin of each neighbour
        sum_=grid[i-1][j]+grid[i+1-size][j]+grid[i][j-1]+grid[i][j+1-size]
        delta = 2.0 * (grid[i][j]*2-1)*(2*sum_-4+mew)
        #get a random number for our boltzmann update
        n = xoroshiro128p_uniform_float32(rng_states, idx)  
        #lock3 is our decision to perform the ising update according to the boltzmann distribution
        lock3=int(1-n < math.exp(-JB*delta))
        #update is zero if the thread or it's neighbours were already locked
        update=lock1*lock2*lock3*(1-2*grid[i][j])
        #add 0,1, or -1 depending on if the cell should be updated (boltzmann and locking)
        cuda.atomic.add(grid[i], j, update)
        cuda.threadfence()
        #make sure the update is visible to the other blocks
        cuda.atomic.add(active[i],j,-lock1)
        #unlock
        cuda.threadfence()
        #make sure the unlocking is visible to the other blocks


"""
# Helper Functions for a potential ising gym
These collectively take the grid and output image observations.

In order to work with the default nature CNN, images must have datatype uint8. Everything
here is adjusted to fit into that. I figured out how to use a custom cnn without this 
limitation so this may change in the future.
"""


@cuda.jit
def E_S(grid,store,s):
    """Calculate total energy and magnetization"""
    i,j=cuda.grid(2)
    if i>=s or j>=s:return
    v=-2*(grid[i][j]*2-1)*(grid[i-1][j]+grid[i][j-1]-1)
    cuda.atomic.add(store,0,v)
    cuda.atomic.add(store,1,grid[i][j])

#just run this with a single thread
@cuda.jit
def set_store(store,time,JB,mew,targE,targM,s):
    """This just sets some of the heuristics in the gpu array store.
    this array will be in gpu memory so it is faster to call a kernel to set the values
    then to set them in normal code.
    """
    store[2]=time
    store[3]=store[1]*255/s**2
    store[4]=JB*50
    store[5]=(mew+4)*32-0.1
    store[6]=256-targE*128
    store[7]=128+targM*128
@cuda.jit
def zero(store):
    """Again this is just faster since store will be in gpu memory"""
    store[0]=0
    store[1]=0
@cuda.jit
def toimg(grid,outarr,store,sin,sout):
    """This converts the grid+heuristics into a uint8 3-channel image
    The first channel gives the ising grid, the second channel gives some heuristics,
    and the third channel gives the target energy and magnetization."""
    #assiming it is square because it saves implementation time
    #also using periodic boundaries lets go
    i,j=cuda.grid(2)
    if (i<sout and j<sout):
        scale=sin//sout
        scl2=scale//2
        sum_=0
        I=i*scale-scl2
        J=j*scale-scl2
        for a in range(scale):
            for b in range(scale):
                sum_+=grid[(I+a)%sin][(J+b)%sin]
                
        outarr[0][i][j]=int(255*sum_/scale**2)    
        
        idx= ((2*i)//sout)*2+(2*j)//sout
        outarr[1][i][j]=int(store[2+idx])
        #store should contain:
        #E,M,Time,M/cell,Temp,Mew,target E/cell, target M/cell in that order.
        outarr[2][i][j]=int(store[6+(2*j)//sout])

class Ising(Thread):
    def __init__(self, N):
        #setup the lattice and helper data structures
        self.N=N
        self.grid=np.asarray(np.random.random([N,N])>0.5,dtype=np.int64)
        self.threadsperblock = (16, 16)#should end up a multiple of 32 I think
        blockspergrid_x = int(np.ceil(self.grid.shape[0] / self.threadsperblock[0]))
        blockspergrid_y = int(np.ceil(self.grid.shape[1] / self.threadsperblock[1]))
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)
        self.blockalt=(min(blockspergrid_x//2,8), min(blockspergrid_y//2,8))
        self.isalive=True
        self.rng_states = create_xoroshiro128p_states(self.grid.size, seed=1)
        self.grid_global_mem = cuda.to_device(self.grid)
        self.JB=5
        self.mew=0
        self.rspeed=100
        self.act=cuda.to_device(np.zeros_like(self.grid,dtype=np.int32))
        super(Ising, self).__init__()
    def fps(self):
        """Gives the number of updates performed in one second"""
        iold = self.index
        time.sleep(1)
        return self.index-iold
    def run(self):
        """Run the ising updates on a thread"""
        self.index=0
        while self.isalive:
            self.index+=1
            if True:
                continuous_ising[self.blockalt, (16,16)](self.grid_global_mem,self.act,50,self.JB,self.mew,self.rng_states,self.N)
                #time.sleep(0.01)
            else:
                fast_ising[self.blockspergrid, self.threadsperblock](self.grid_global_mem,self.JB,self.mew,0,self.rng_states,self.N)
                fast_ising[self.blockspergrid, self.threadsperblock](self.grid_global_mem,self.JB,self.mew,1,self.rng_states,self.N)
    def step(self,nsteps=50):
        continuous_ising[self.blockalt, (16,16)](self.grid_global_mem,self.act,nsteps,self.JB,self.mew,self.rng_states,self.N)
