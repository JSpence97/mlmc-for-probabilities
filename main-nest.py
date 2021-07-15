import numpy as np
from samplers import nested_model_prob
from mlmc import mlmc

np.random.seed(1)


# Sampler parameters
names = ['ad', 'ad-nestsim', 'st-gam1', 'st-gam2']  # Method names to identify saved files
gammas = [1, 1, 1, 2]  # Cost refinement factr
betas = [1, 1, 0.5, 1]  # Corresponding variance reduction rate
alphas = [2, 2, 1, 2]  # Corresponding bias reduction rate
rs = [1.95, 1.95, -1, -1]  # r-values for adaptive sampling (-1 for non-adaptive levels)
reuse = [True, False, True, True]
err_split = 0.75  # Proportion theta of cost attributed to statistical error
p = 0.05  # ~~(1-p)% confidence in TOL
N0 = 32 # No. inner samples at level 0
ells = np.arange(9)  # Levels to independently compute multilevel statistics
tols = np.logspace(-1, -2.5, 10)*0.025 # Tolerances
tmax = 24*3600  # Max runtime per sampler
paramrange = 3  # No. previous levels to compute bayesian estimates of constants
M = 10**5  # Number samples to compute level ell statistics
M0 = 10**5 # Number samples to approximate optimal starting level
c = 3/np.sqrt(N0) # Confidence constant to match that in [3] Giles, Haji-Ali 'Multilevel nested simulation for efficient risk estimation', 2018.
L_eta = 0.0805  # Loss threshold for the nested simulation model problem to ensure prob(g>0) = 0.025



# Define \sigma_\ell as the sample variance computed from N_{\ell} samples
def sigma(self, noise, ell, eta):
    num_pts = self.N0*2**(self.gamma*(ell+eta))  # Number of points at refined level
    indices = 2**(self.gamma*(ell-self.ell_m1 + eta))  # No. blocks of N_{\ell-1} samples to use
    return np.sqrt(np.sum(noise.Xsq[0:indices, :], axis = 0)/num_pts \
        - (np.sum(noise.X[0:indices, :], axis = 0)/num_pts)**2)

# Run experiments defined above and save data
counter = -1
for name in names:
    counter += 1
    print('\n\n---------------------------------------------------------\n\n', name)
    sampler = nested_model_prob(N0=N0, gamma=gammas[counter], sigma=sigma, reuse_samples=reuse[counter], loss = L_eta)
    mlmc_temp = mlmc(r=rs[counter], c=c, sampler = sampler)
    for ell in ells:  # Pre compute level ell statistics for required levels for independent results
        print('\nell = ', ell)
        mlmc_temp.evaluate(ell, M)
        print('\ncost: ', mlmc_temp.output.costs)
        print('variance: ', mlmc_temp.output.Vs)
        print('means: ', mlmc_temp.output.means)
    mlmc_temp.save(name + '-levels.csv')  # Save result of independent experiment
    mlmc_temp.find_l0(M0, tol_ell0 = 1.1)    # Find optimal starting level
    mlmc_temp.cmlmc(tols, beta = betas[counter], alpha = alphas[counter], err_split = err_split, p = p, tmax = tmax, \
        paramrange = paramrange) # Perform mlmc  and save data
    mlmc_temp.save_mlmc(name + '-mlmc.csv')
