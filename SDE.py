import numpy as np

# Create SDE class

class SDE:
    # This class aims to model the Stochastic Differential Equation
    # X_t = drift(X_t,t)dt + diffuse(X_t,t)dW_t
    # 
    # X_t is n-dimensional process (i.e. lives in R^n)
    # W_t is an m-dimensional Wiener process (Brownian motion)
    # drift: R^n-> R^n
    # diffuse: R^n-> R^{nxm}    
    
    def __init__(self, dim, dimW, drift, diffuse):
        # All these need to be compatible somehow and should handle errors
        self.dim = dim                  # dimension of process X (>0)
        self.dimW = dimW                # dimension of Wiener process W (>0)
        self.drift = drift              # function of position and time
        self.diffuse = diffuse          # function of position and time
    
    # This uses Euler-Maruyama (EM) to solve the SDE defined by self
    def EMsolve(self, Nsim=10, N=1000, T=1, x0=None):
        if x0 is None:
            x0 = np.zeros(self.dim)     # initialize to zero-vector
        
        # Nsim is number of simulated solutions
        # N is number of time-steps/time-intervals: t0,t1,...,tN
        # T is time at which simulation path terminates
        # x0 is initial position (vector of same dimension as X_t)
        
        n = self.dim                 # dimension of process
        m = self.dimW                # dimension of noise dW in Ito integral
        drift = self.drift           # drift term (in front of dt)
        diffuse = self.diffuse       # diffusion term (in front of dW)
            
        dt = T/(N-1)                    # time step size
        #t = np.arange(0,T+dt,dt)    # array of times we are sampling
        t = np.linspace(0,T,N)    # array of times we are sampling
        X = np.zeros((N,n))       # initialize: n-dim, N-samples 
        
        simPaths = [ None ]*Nsim            # init list of simulated solns
        X[0,] = x0                          # initial point
        for iSim in range(Nsim):
            for i in range(0,t.size-1):
                # sample m-dimensional noise
                dW = np.random.normal(0,np.sqrt(dt),m)
                # generate solution at next time
                X[i+1,] = X[i,] + drift(X[i,],t[i])*dt + diffuse(X[i,],t[i]).dot(dW)
            
            # Store solution in simPaths. W/o deepcopy list is overwritten
            simPaths[iSim] = np.column_stack( (t,X) )
            
        # return time girdpoints and list of simulated solutions
        return simPaths
 
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    
    # Define Geometric Brownian Motion Class
    class SDE_GeomBM(SDE):
        def __init__(self,mu=1, sigma=0.1):
            self.args = {'mu': mu, 'sigma': sigma}
            
            # Define drift and diffusion
            drift   = lambda x,t: mu*x
            diffuse = lambda x,t: np.array([sigma])*x
            
            super().__init__(1, 1, drift, diffuse)
            
        def print_params(self):
            for kw, val in self.args.items():
                print (kw, ':', val, end='\n')

    # Define (Langevin) Spiral Brownian Motion Class
    class SDE_SpiralBM(SDE):
        def __init__(self,alpha=1, delta=0.1, gamma=1, KBT=0.0001):
            self.args = {'alpha': alpha, 'delta': delta, 
                         'gamma': gamma, 'KBT': KBT}
            
            # Compute structural constants
            self.D = KBT/gamma
            self.A = (1/gamma)*np.array([[-delta, alpha],[-alpha, -delta]])
            
            # Define drift and diffusion
            drift   = lambda x,t: (self.A).dot(x)
            diffuse = lambda x,t: np.sqrt(2*self.D)*np.eye(2)
    
            super().__init__(2, 2, drift, diffuse)
            
        def print_params(self):
            for kw, val in self.args.items():
                print (kw, ':', val, end='\n')
            print('\nD:  ', self.D, '\nA:\n',  self.A)


    # Make and Record trajectories
    start = time.time()
    params = {'mu':0.5, 'sigma':0.5}
    gbm = SDE_GeomBM(**params)
    gbm_solns = gbm.EMsolve(Nsim=15, N=1000, T=5, x0=np.array([1]) ) 
    print("Elapsed time (Geometric BM): ", time.time()-start)

    start = time.time()
    params = {'alpha':1, 'delta':0.1, 'gamma':1, 'KBT':0.0015}
    spiralbm = SDE_SpiralBM(**params)
    spiralbm_solns = spiralbm.EMsolve(Nsim=10, N=10000, T=8, x0=np.array([0,1]) ) 
    print("Elapsed time (Spiral BM): ", time.time()-start)

    # Plot geometric brownian motion trajectories
    for i in range(len(gbm_solns)):
        plt.plot(gbm_solns[i][:,0],gbm_solns[i][:,1])
    plt.title('Geometric Brownian Motion')
    plt.show() 
    gbm.print_params()   
    
    # Plot spiral brownian motion trajectories
    for i in range(len(spiralbm_solns)):
        plt.plot(spiralbm_solns[i][:,1],spiralbm_solns[i][:,2])
    plt.title('Spiral Langevin Process')
    plt.show() 
    spiralbm.print_params()
    

    

