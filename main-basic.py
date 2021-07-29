import numpy as np
from numpy.random import randn
from samplers import basic_sampler
from scipy.stats import norm
from mlmc import mlmc

np.random.seed(1)


# Options
sigma = np.sqrt(3)  # Constant value for sigma
L_eta = norm.ppf(0.975)  # Loss threshold to give prob(g>0) = 0.025
noise_samples = lambda M: randn(2,M)  # First row for Z, second row for noise realisation
Zhat = lambda M, ell, noise, beta: 2**(-beta*ell/2)*(2**(-beta*ell/2) + noise[1,:]**2 - 1)

# Sampler params
rs = [1.95, -1, -1]  # r parameter for adaptive sampling (-1 for non-adaptive levels)
gammas = [1, 1, 2]  # Cost refinement factor between levels
betas = [1, 1, 2]  # variance reduction rate
alphas = betas  # Follows from model of Zhat above
names = ['ad', 'st-gam1', 'st-gam2']  # Names for saved data files
ells = np.arange(9)
M = 10**7  # Number of samples for sampling level statistics
c = 1 # Confidence constant

# mlmc params
tols = np.logspace(-1, -2, 10)*0.025
err_split = 0.75
paramrange = 3

# Run experiments detailed above and save data
counter = -1
for name in names:
    counter += 1
    print('--------------------------------------\n', name)
    sampler = basic_sampler(gamma = gammas[counter], sigma = sigma,loss = L_eta, noise_samples = noise_samples,\
        Zhat = lambda M, ell, noise: Zhat(M, ell, noise, betas[counter]))
    mlmc_temp = mlmc(r=rs[counter], c=c, sampler = sampler)
    mlmc_temp.evaluate(ells, M)  # Independent calculation for level ell statistcs
    mlmc_temp.save(name + '-levels.csv')  # Save independent experiments
    mlmc_temp.find_l0(M, tol_ell0=1.1) # Find optimal starting level then perform and store mlmc results
    mlmc_temp.cmlmc(tols,beta =  betas[counter], alpha = alphas[counter], err_split = err_split, paramrange = paramrange)
    mlmc_temp.save_mlmc(name + '-mlmc.csv')
