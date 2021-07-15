# This file contains classes containing necessary methods used to sample g to compute expectations of the form
#           E[H(g)].
# The following problems are defined:
# SDE discretization for Digital Options
# Nested simulation
# A basic sampler with artificial cost

# Standard imports
import numpy as np
from numpy.random import randn, standard_t, rand
from scipy.stats import norm

## Digital Option
class digital_option:
    """
    Base class for digital options with
        dS_t = a(t, S_t)dt + b(t, S_t)dW_t
    up to maturity T, where S_t is `num_stocks` dimensional. Considers the option with payoff: combine_sde(S_T) - threshold.
    """
    def __init__(self, a, b, T, S0, combine_sde, gamma, steps0, sigma, num_stocks = 1, \
        threshold = 0, corr  = 0, method = 'euler', db = lambda t, S: 0):

        self.a = a # Drift
        self.b = b  # Diffusion
        self.T = T # Maturity
        self.S0 = S0 # Initial value
        self.gamma = gamma # cost rate
        self.steps0 = steps0  # Steps at level 0
        self.sigma = sigma  # Sigma_ell (constant here)
        self.num_stocks = num_stocks
        self.num_bm = num_stocks # Number of independent Brownian motions required
        self.threshold = threshold  # Threshold for unit returns
        self.corr = corr  # Correlation coefficient of noise
        if abs(corr) > 1e-15 and self.num_stocks > 1.5:  # If nontrivial correlation, add another Brownian process
            self.num_bm += 1
        self.combine_sde = lambda g: combine_sde(g)  # Final payoff is combine_sde(S_T) - threshold
        self.method = method  # Allows for E-M/Milstein (note: Milstein currently only works in 1D).
        self.db = lambda t, S: db(t, S)  # Derivative of diffusion coefficient term (only needed for Milstein)


    def init_ell(self, ell, ell0):  # Initialise parameters specific to level ell
        self.ell = ell
        if ell == ell0:
            self.ell_m1 = ell
        else:
            self.ell_m1 = ell - 1

    def sample_noise(self, M): # Returns M sample paths for each of the num_bm Brownian motions using
                               # steps0*2**(gamma*ell) timesteps
        W = np.zeros((self.steps0*2**(self.gamma*self.ell)+1, self.num_bm,  M))
        W[1:, :] = np.cumsum(np.sqrt(self.T/(W.shape[0] - 1))*randn(W.shape[0] - 1, self.num_bm, M), axis = 0)
        return W

    def sample_g(self, noise, ell, eta): # Sample Euler-Maruyama approximation given noise
        steps = self.steps0*2**(self.gamma*(ell+eta))
        diff = int((noise.shape[0] - 1)/steps)  # Number intermediate brownian points to skip at level ell + eta

        if self.corr > 1e-15 and self.num_stocks >1.5:  # Add correlation factor if present
            BM = self.corr*noise[:,-1,:].reshape(noise.shape[0],1, noise.shape[-1]) + np.sqrt(1-self.corr**2)*noise[:,:-1, :]
        else:
            BM = noise

        Sn = np.ones((self.num_stocks, noise.shape[-1]))*self.S0  # Initial values
        t = 0
        dt = self.T/steps  # step-size

        for n in range(steps): # Loop over all timesteps
            if self.method == 'milstein':  # Milstein step if necessary (presently only works in 1D)
                Sn += self.a(t, Sn)*dt + self.b(t, Sn)*(BM[(n+1)*diff,:, :] - BM[n*diff,:, :]) + \
                    0.5*self.b(t, Sn)*self.db(t, Sn)*((BM[(n+1)*diff,:, :] - BM[n*diff,:, :])**2 - dt)
            else:  # Else, Euler-Maruyama step
                Sn += self.a(t, Sn)*dt + self.b(t, Sn)*(BM[(n+1)*diff,:, :] - BM[n*diff,:, :])
            t += dt
        return self.combine_sde(Sn).reshape((1, Sn.shape[-1])) - self.threshold  # Return payoff of digital option

    def split_noise(self, noise, done):  # Split accepted/rejected Brownian paths
        return (noise[:, :, done], noise[:, :, done == False])

    def refine_noise(self, noise, ell, eta):  # Brownian Bridge refinement of Brownian paths
        dt = self.T/(noise.shape[0] - 1)
        for i in range(self.gamma):  # Refine step-size by factor 1/2, gamma times
            oldsteps = 2*np.arange(int(noise.shape[0]))
            newnoise = np.zeros((1 + 2*(noise.shape[0]-1), self.num_bm, noise.shape[-1]))
            newnoise[oldsteps, :, :] = noise
            newnoise[oldsteps[:-1] + 1, :, :] = 0.5*(noise[:-1, :, :] + noise[1:, :, :]) \
                + np.sqrt(dt)/2 * randn(noise.shape[0] - 1, self.num_bm, noise.shape[-1])
            noise = newnoise
        return noise

    def delta(self, gell, noise, ell, eta):  # Delta variable with constant sigma_\ell = sigma
        return np.abs(gell[0,:])/self.sigma

    # Return multilevel correction and costs at fine /coarse level
    def multilevel_correction(self, gfine, gcoarse, ellf, ellc, noise):
        if type(gcoarse) == str:  # If we dont require coarse samples
            return (gfine > 0).astype(int), gfine.shape[1]*self.steps0*2**(self.gamma*ellf),\
                gfine.shape[1]*self.steps0*2**(self.gamma*ellf)
        else:
            fine = (gfine > 0).astype(int)
            return fine - (gcoarse > 0).astype(int), fine,\
                gfine.shape[1]*self.steps0*(2**(self.gamma*ellf) + 2**(self.gamma*ellc)),\
                gfine.shape[1]*self.steps0*2**(self.gamma*ellf)


## Nested Expectation - g = \E{X|Y}; noise = (Y, \{X(Y)\}_i). Approximates g by inner MC average
## g_\ell = N_\ell^{-1}\sum_{1<=n<=N_\ell} X^{(n)}(Y), uses antithetic sampling of the multilevel correction term as in
##  [3] Giles, Haji-Ali 'Multilevel nested simulation for efficient risk estimation', 2018.
class noise_nested:  # Stores noise - samples Y, X, X^2. Note: We need X^2 to compute the sample variance for sigma_\ell
    def __init__(self, Y, X, Xsq):
        self.Y = Y  # Initial outer samples
        self.X = X  # Inner samples (shared between levels)
        self.Xsq = Xsq  # Squared inner samples (shared between levels)
        self.size = Y.size  # Used to determine no. remaining samples when adaptively sampling

class nested_base:
    def __init__(self, N0 = 1, gamma = 1, sigma = lambda self, noise, ell, eta: 1, reuse_samples = True, loss = 0):
        self.N0 = N0  # Number of samples for the inner MC at level 0
        self.gamma = gamma  # Refinement rate
        self.sigma = sigma  # Used in \delta_\ell to measure sample dependent variability (i.e. conditional sample s.d.)
        self.N = lambda ell: int(self.N0*2.**(self.gamma*ell))  # Returns no. inner samples given ell
        self.reuse = reuse_samples  # If False, uses adaptive nested simulation algorithm [3, Algorithm 1]
        self.loss = loss  # Constant loss threshold

    def sample_y(self, M):
        raise Exception("Function 'sample_y' has not been defined.")

    def sample_x(self, Y, ell):
        raise Exception("Function 'sample_x' has not been defined.")

    def init_ell(self, ell, ell0):  # Initialise parameters specific to level ell
        self.ell = ell
        if ell == ell0:
            self.ell_m1 = ell
        else:
            self.ell_m1 = ell - 1

    def sample_noise(self, M):  # Initial samples of X, Y at level ell, store X in blocks of sums of N02^{gamma(l-1)} terms
                                # to save memory
        if self.reuse == True:  # If using the general adaptive sampler
            Y = self.sample_y(M)  # Compute Yvals
            if self.ell > self.ell_m1: # Test whether ell > ell - 1 to determine number of blocks required
                num_divs = 2**self.gamma
            else:
                num_divs = 1
            # Compute X and Xsq as blocks of sums of size 2^{gamma*(ell-1)}
            X = np.zeros((num_divs, M))
            Xsq = np.zeros((num_divs, M))
            for i in range(num_divs):
                x = self.sample_x(Y, self.ell_m1)
                X[i, :] = np.sum(x, axis = 0)
                Xsq[i, :] = np.sum(x**2, axis = 0)

            return noise_nested(Y, X, Xsq)

        else:
            return self.sample_y(M)  # For [3], only sample Y as X will be resampled later

    def split_noise(self, noise, done): # Splits noise into accepted/rejected samples according to method determined by reuse
        if self.reuse == True:
            noise_acc = noise_nested(noise.Y[done == True], noise.X[:, done == True], noise.Xsq[:, done == True])
            noise_rej = noise_nested(noise.Y[done == False], noise.X[:, done == False], noise.Xsq[:, done == False])
            return noise_acc, noise_rej
        else:
            return noise[done==True], noise[done==False]


    def refine_noise(self, noise, ell, eta):
        if self.reuse == True: # For the general adaptive algorthm, refine the inner MC algorithm by adding
                               # (2**(gamma) - 1) times more samples of X
            newRows = 2**(self.gamma*(ell - self.ell_m1 + eta))*(2**self.gamma - 1)  # Number blocks of N_{\ell-1} samples
            XNew = np.zeros((newRows, noise.size))
            XsqNew = np.zeros((newRows, noise.size))

            for i in range(newRows):  # Sum to sample each required block
                x = self.sample_x(noise.Y, self.ell_m1)
                XNew[i, :] = np.sum(x, axis = 0)
                XsqNew[i, :] = np.sum(x**2, axis = 0)
            noise.X = np.concatenate([noise.X, XNew], axis = 0)
            noise.Xsq = np.concatenate([noise.Xsq, XsqNew], axis = 0)
            return noise

        else:
            return noise  # Not needed in [3] since we resample all X terms

    def sample_g(self, noise, ell, eta): # Compute antithetic means for g
        if self.reuse == True:  # General adaptive sampling
            num_sum = 2**(self.gamma*(ell - self.ell_m1 + eta))  # No. rows of noise.X to use for each mean
            return np.array([np.sum(noise.X[i*num_sum:(i+1)*num_sum, :], axis = 0)/self.N(ell + eta)\
                for i in range(int(noise.X.shape[0]/num_sum))]) - self.loss

        else:  # As in [3]
            # Computes mean of first and second moments of X from samples at level ell+eta
            x_samples = self.sample_x(noise, ell+eta)
            # Return g_{\ell+\eta} in first row, \sigma_{|ell+\eta} in the second and the level in the third
            # to feed information to adaptive sampler
            return np.array((np.mean(x_samples, axis = 0) - self.loss, np.std(x_samples, axis = 0)))

    def multilevel_correction(self, g_fine, g_coarse, ellf, ellc, noise):
     # Returns antithetic multilevel correction term and associated sampling costs
        if self.reuse == True:
            if type(g_coarse) == str: # If we dont require coarse samples
                return np.mean(g_fine > 0, axis = 0), g_fine.shape[1]*self.N(ellf),  g_fine.shape[1]*self.N(ellf)
            else:
                fine = np.mean(g_fine > 0, axis = 0)
                return fine - np.mean(g_coarse > 0, axis = 0), fine, g_fine.shape[1]*self.N(ellf), g_fine.shape[1]*self.N(ellf)

        else:
            if type(g_coarse) == str: # If we dont require coarse samples
                return np.mean(self.sample_x(noise, ellf), axis = 0) > self.loss,\
                    noise.size*(np.sum([self.N(self.ell+i)for i in range(ellf - self.ell + 1)])+self.N(ellf)),\
                    noise.size*(np.sum([self.N(self.ell+i)for i in range(ellf - self.ell + 1)]) + self.N(ellf))
            else:
                x_samples = self.sample_x(noise, max(ellf, ellc))
                Nmax = int(self.N(max(ellf, ellc)))
                fine = np.mean([np.mean(x_samples[i*self.N(ellf):(i+1)*self.N(ellf),:], axis=0)  > self.loss\
                    for i in range(int(Nmax/self.N(ellf)))], axis = 0)   # Antithetic fine estimator
                coarse = np.mean([np.mean(x_samples[i*self.N(ellc):(i+1)*self.N(ellc),:], axis=0)  > self.loss \
                for i in range(int(Nmax/self.N(ellc)))], axis = 0)  # Antithetic coarse estimator
                # Return antithetic mean and costs
                return fine - coarse, fine,\
                    noise.size*(np.sum([self.N(self.ell+i)for i in range(ellf - self.ell + 1)])\
                            + np.sum([self.N(self.ell_m1 + i) for i in range(ellc - self.ell_m1 + 1)]) + Nmax),\
                    noise.size*(np.sum([self.N(self.ell+i)for i in range(ellf - self.ell + 1)]) +self.N(ellf))

    def delta(self, g, noise, ell, eta):
        if self.reuse == True:  # Delta with variable sigma
            return np.abs(g[0, :])/self.sigma(self, noise, ell, eta)  # Must use only a single realisation of g_\ell here
                                                                      # (hence row 0 of g) otherwise we break the
                                                                      # cancellation required for the teloscopic sum in MLMC.
        else:  # Delta using sigma as the sample standard deviation as in [3]
            return np.abs(g[0,:])/g[1,:]

class nested_model_prob(nested_base):
    # Problem specific parameters
    tau = 0.02
    def sample_y(self, M):  # Return standard normal RV for Y
        return randn(M)

    def sample_x(self, Y, ell):  # Return values of X given Y
        Ntemp = self.N(ell)
        return self.tau*(Y**2 - randn(Ntemp, Y.size)**2) \
            + 2*np.sqrt(self.tau*(1-self.tau))*Y*randn(Ntemp, Y.size)

################################################################################

## Basic sampler where g = Z, g_\ell = Z + \hat Z
class basic_sampler:
    def __init__(self, gamma=1, sigma=1, loss=1, noise_samples = lambda M: randn(M), \
            Zhat = lambda M, ell, noise: 2**(-ell/2)*randn(M)):
        self.gamma = gamma  # Cost scaling rate
        self.sigma = sigma  # Constant sigma term
        self.loss = loss  # Loss threshold
        self.noise_samples = lambda M: noise_samples(M)  # First row of noise_samples(M) must denote the true solution
        self.Zhat = lambda M, ell, noise: Zhat(M, ell, noise)

    def init_ell(self, ell, ell0):  # Initialise parameters specific to level ell
        self.ell = ell
        if ell == ell0:
            self.ell_m1 = ell
        else:
            self.ell_m1 = ell - 1

    def sample_noise(self, M):  # Initial noise as supplied by input term
        return self.noise_samples(M)

    def split_noise(self, noise, done):
        return noise[:,done], noise[:,done==False]

    def refine_noise(self, noise, ell, eta):
        return noise # Not needed

    def sample_g(self, noise, ell, eta):  # Sample g_\ell
        return (noise[0,:] + self.Zhat(noise.shape[1], ell+eta, noise)).reshape((1, noise.shape[1]))  - self.loss

    def multilevel_correction(self, gfine, gcoarse, ellf, ellc, noise):  # Return multilevel correction and sampling cost
        if type(gcoarse)==str:
            return (gfine > 0).astype(int), 2**(self.gamma*ellf)*gfine.shape[1],\
                2**(self.gamma*ellf)*gfine.shape[1]
        else:
            fine = (gfine > 0).astype(int)
            return fine - (gcoarse > 0).astype(int),fine,\
                (2**(self.gamma*ellf)+2**(self.gamma*ellc))*gfine.shape[1],   2**(self.gamma*ellf)*gfine.shape[1]

    def delta(self, g, noise, ell, eta):  # Compute delta(g)
        return np.abs(g[0,:])/self.sigma
