import numpy as np
from numpy.random import randn, rand
from samplers import digital_option
from scipy.stats import norm
from mlmc import mlmc

np.random.seed(1)
# Problem parameters
T = 1  # Maturity
######## Uncomment for ND problem, comment for 1D problem
num_stocks = 10
S0 = 0.9 + 0.2*rand(num_stocks, 1)  # Initial values
mu = 0.05 + 0.1*rand(num_stocks, 1)  # Drifts
vol = 0.01 + 0.39*rand(num_stocks, 1)  # Volatilities
loss_thresh = 1.2857
rho = 0.2
methods = ['euler', 'euler', 'euler']  # Use Euler method only (Milstein not yet implemented for multidimensional problems here)
rs = [1.95, -1, -1]  # r value to use per iteration (-1 for non-adaptive levels)
gammas = [1, 1, 2]  # Cost scaling to use per iteration
alphas = [2, 1, 2]  # Corresponding bias reduction rates from theory
betas = [1, 0.5, 1]  # Corresponding variance reduction rates from theory
names = ['ad-d10', 'st-gam1-d10', 'st-gam2-d10']
########

######## Uncomment for 1D problem, comment for ND problem
# num_stocks = 1
# S0  = 1  # Initial value
# mu = 0.05  # Drift
# vol =0.4  # Volatility
# loss_thresh = np.sum(S0*np.exp(-vol*np.sqrt(T)*norm.ppf(0.025) - (vol**2/2 - mu)*T))
# rho = 0
# methods = ['euler',  'euler', 'milstein',  'milstein']  # SDE simulation schemes to use
# rs = [1.95, -1, 10, -1]  # r value to use per iteration
# gammas = [1, 2, 1, 2]  # Cost scaling to use per iteration
# alphas = [2, 2, 2, 2]  # Corresponding bias reduction rates from theory
# betas = [1, 1, 2, 2]  # Corresponding variance reduction rates from theory
# names = ['ad', 'st-gam2', 'ad', 'st-gam2']
########

a = lambda t, S: mu*S  # Drift
b = lambda t, S: vol*S  # Diffusion
db = lambda t, S: vol
combine_sde = lambda g: np.sum(g, axis =0)/g.shape[0]  # Payoff to use to evaluate option


# Sampler params
tmax = 0.1*3600  # Max runtime
paramrange = 3  # No. previous levels used in bayesian estimate for constants
err_split = 0.75
p = 0.05  # Desired prob of observing worse error bound
ells = np.arange(7)  # Levels to pre-compute mlmc statistics separate from MLMC computation
M = 10**6  # Number samples for sampling level statistics
M0 = 10**5  # Number samples to approximate optimal l0
tols = np.logspace(-1, -4, 10)*0.025 # Tolerances
steps0 = 1  # Number of steps at level 0
sigma = 1/np.sqrt(num_stocks)  # Noise level in \delta_{\ell}
c = 1  # Confidence constant

# Run experiments detailed above and save data
counter = -1
for name in names:  # For each method defined above
    counter += 1
    print('--------------------------------------\n', name, ' - ', methods[counter])
    sampler = digital_option(a, b, T, S0, combine_sde, gammas[counter], steps0, sigma, num_stocks = num_stocks,\
        corr = rho, threshold = loss_thresh, method = methods[counter], db = db)  # Define digital option sampler
    mlmc_temp = mlmc(r=rs[counter], c=c, sampler = sampler)  # Define corresonding mlmc estimator
    mlmc_temp.evaluate(ells, M) # Loop over all levels to pre-compute statistics independent from main MLMC computation
    mlmc_temp.save(name + '-' + methods[counter]+ '-levels.csv') # Save data
    mlmc_temp.find_l0(M0, tol_ell0 = 1.1)  # Estimate optimal starting level
    mlmc_temp.cmlmc(tols, beta = betas[counter], alpha = alphas[counter], err_split = err_split, p = p,\
        tmax = tmax, paramrange = paramrange)  # MLMC computation over tolerances in tols
    mlmc_temp.save_mlmc( name + '-' + methods[counter] + '-mlmc.csv')  # Save data
