 # Adaptive Multilevel Monte Carlo for Probabilities

 This repository contains code used to produce results in the following paper:

 [1]: A-L. Haji-Ali, J. Spence and A. Teckentrup. "Adaptive Multilevel Monte Carlo for Probabilities". 2021.

This work considers the problem of computing probabilities of the form `P[G\in\Omega]`, where the random variable `G`
requires approximate simulation. In [1], we reduce such problems to the form `P[g>0] = E[H(g)]` for a one dimensional variable
`g`, where `H` is the Heaviside function. The repository is organised as follows:

### `adaptive_sampling.py`:
This file contains the function `adaptive_sampler`  which samples the MLMC correction term with adaptive refinement as in [1, Algorithm 1]:

                                      H(g_{ell + eta_ell}) - H(g_{ell - 1 + eta_{ell-1}}).

This function must be supplied a `sampler` class which must be defined separately, classes for the problems considered in [1] are defined in `samplers.py`.
The supplied class must contain the following methods:
  - `init_ell` <- initialises any level dependent parameters of the problem.
  - `sample_noise` <- samples initial problem-specific noise required to sample `g_ell`.
  - `sample_g(noise, ell, eta)` <- approximates g at level `ell+eta` using the supplied noise.
  - `delta(g, ell, eta)` <- computes the variable `delta(g_{ell+eta})` defined in [1] used to determine whether samples should be refined further.
  - `split_noise` <- splits the noise according to accepted/rejected samples at the current level.
  - `refine_noise` <- refines supplied nise to next level of approximation.
  - `multilevel_correction` <- samples the term \Delta H_ell as defined in [1]. By specifying this term directly, we allow for modifications to the multilevel correction term, for example using antithetic averages of the Heavisides for the nested simulation problem as in [3] to improve the total required work by a constant.

This file also contains the function `det_sampler` which serves the same purpose as `adaptive_sampler` but with deterministic
sampling of levels (that is, sampling `H(g_ell) - H(g_{ell-1})`).

### `samplers.py`:
Contains `sampler` classes to be supplied to the functions in `adaptive_sampling.py` for the following problems considered in [1]:
- Digital option problem, where `g` is a functional of a d-dimensional SDE to be approximated by the Euler-Maruyama scheme, or a 1-dimensional SDE to be approximated by the Euler-Maruyama or Milstein schemes.
-   The nested simulation problem where `g = E[X|Y]` is approximted by an inner Monte Carlo average  as considered in [3]. This class contains an optional parameter `reuse_samples` (default True). If False, this parameter reverts the sampler to the adaptive sampling method used in [3] which must re-sample all terms used in the inner Monte Carlo average when adaptively refining between levels. Otherwise, the method in [1] is used, where the inner Monte Carlo samples can be reused for the refined levels.
- The artificial MLMC problem, which is used in [1, Appendix A] to test certain aspects of the theory.

### `mlmc.py`:
Contains class `mlmc` which defines a specific MLMC estimator given a `sampler` class. Depending on the sign of the input variable `r`, either adaptive or deterministic levels are used for the multilevel correction term, with the corresponding function taken from `adaptive_sampling.py`. Has methods to compute and store the level `ell` MLMC statistics separate from the main MLMC computation for the purpose of verifying the theoretical bounds in [1]. The method `find_l0` estimates the optimal starting level for the MLMC computation as discussed in [3]. MLMC estimates are returned for a specified set of tolerances using the method `cmlmc`, which uses an approach adapted from [2] to form estimates for the mean square error made to determine convergence.

## Usage:
The files `main_x.py` combine the files discussed above to compute the numerical experiments considered in [1]:
 - `main_sde.py` contains parameters for the 1 and 10 dimensional digital option problems considered in [1]. The file approximates the level `ell` statistics as well as computing MLMC estimates to specified error tolerances for several adaptive and non-adaptive multilevel samplers, returning the results obtained in [1, Section 4.2]. To switch between the 1 and 10 dimensional problem, one should comment and uncomment the relevant marked code at the top of this file.
 - `main_nest.py` contains code to reproduce the results for the nested simulation model problem considered in [1, Section 4.1] and [3].
 - `main_basic.py` holds the code used to test the adaptive bias convergence rate under weaker conditions, producing the results in [1, Appendix A].




## Additional References
[2]: N. Collier, A-L. Haji-Ali, F. Nobile et. al. "A continuation multilevel Monte Carlo algorithm". 2015.

[3]: M. B. Giles, A-L. Haji-Ali 'Multilevel nested simulation for efficient risk estimation', 2018.
