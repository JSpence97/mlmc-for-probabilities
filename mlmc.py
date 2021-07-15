import numpy as np
from scipy.stats import norm
from time import perf_counter
from adaptive_sampling import adaptive_sampler, det_sampler


class amlmc_out:  # Class to store output data from mlmc
    def __init__(self):
        self.levels = []  # Levels used
        self.costs = []  # Average sampling cost per level
        self.Vfs = []  # Variance of the fine estimators
        self.Vs = []  # Variance of the multilevel correction terms
        self.mfs = []  # Mean of the fine estimators
        self.means = []  # Mean of the multilevel correction terms
        self.sums = []  # Continuous sum for first two moments of multilevel correction terms
        self.sumsf = []  # Continuous sum for first two moments of fine terms
        self.Ms = []  # Number samples used per level

    def update(self, M, ell, cost, sums, sumsf):  # Used to update data due to new terms
        if ell in self.levels: # If ell has already been considered, update existing terms
            index = self.levels.index(ell)
        else: # Otherwise append to existing results
            self.levels.append(ell)
            self.costs.append(0)
            self.Vfs.append(0)
            self.Vs.append(0)
            self.mfs.append(0)
            self.means.append(0)
            self.sums.append(np.zeros(2))
            self.sumsf.append(np.zeros(2))
            self.Ms.append(0)
            index = -1
        self.sums[index] += sums
        self.sumsf[index] += sumsf
        self.costs[index] += cost
        self.Ms[index] += M
        self.Vfs[index] = self.sumsf[index][1]/self.Ms[index] - (self.sumsf[index][0]/self.Ms[index])**2
        self.Vs[index] = self.sums[index][1]/self.Ms[index] - (self.sums[index][0]/self.Ms[index])**2
        self.mfs[index] = self.sumsf[index][0]/self.Ms[index]
        self.means[index] = self.sums[index][0]/self.Ms[index]

class mlmc:
    """
    Class to hold adaptive mlmc level ell sampler, needs to be supplied adaptive rate r (if r <= 0 then uses
    deterministic sampling instead), and sampler class from samplers.py to define problem samples.
    The supplied rate theta (default 1) scales the maximum refined level in adaptive sampling.
    """
    def __init__(self, r = -1, ell0 = 0, c = 1, sampler = None, theta = 1):
        if sampler == None:
            raise Exception("multilevel correction class must be supplied a problem sampler")
        # Declare multilevel correction sampler according to value of r.
        self.ell0 = ell0
        self.gamma = sampler.gamma  # Cost refinement rate
        # Define mlmc correction term sampler based on value of r
        if r > 0:
            self.mlmc_sampler = lambda ell, ell0, M: adaptive_sampler(ell, ell0, M, r, c, sampler, theta)
        else:
            self.mlmc_sampler = lambda ell, ell0, M: det_sampler(ell, ell0, M, sampler)
        self.output = amlmc_out() # Initialise output object


    def evaluate(self, ell, M):
    #Implements level l functional of MLMC for purpose of numerical experiments independent of full MLMC computation
        # Initialise params
        cost = 0
        sums = 0
        sumsf = 0
        Vf = 0  # Variance of fine estimator
        mf = 0  # Mean of fine estimator

        # Perform level l mlmc (scale number of samples taken according to level to save memory costs)
        M_temp = int(min(10**6/(2**(self.gamma*ell)), M))
        count = 0
        while count < M:
            term = self.mlmc_sampler(ell, self.ell0, M_temp)
            sums += term[0]
            sumsf += np.array(term[1:3])
            cost += term[-2]
            count += M_temp
            M_temp = int(min(10**6/(2**(self.gamma*ell)), M - count))
        self.output.update(M, ell, cost, sums, sumsf)



    def mlmc_ell(self, ell, ell0, M):
    #Implements level l functional of mlmc for the MLMC computation in cmlmc defined below
        # Initial params
        cost = 0
        cost_f = 0
        sums = 0
        fine = 0
        fine_sq = 0

        # Perform level l mlmc (scale number of samples taken according to level to save memory costs)
        M_t = int(max(min(10**6/(2**(self.gamma*ell)), M), 1))
        count = 0
        while count < M:
            term = self.mlmc_sampler(ell, self.ell0, M_t)
            sums += term[0]
            cost += term[-2]
            cost_f += term[-1]
            count += M_t
            fine += term[1]
            fine_sq += term[2]
            M_t = int(max(min(10**6/(2**(self.gamma*ell)), M - count), 1))
        return sums, fine, fine_sq, cost, cost_f

    def find_l0(self, M, tol_ell0 = 1): # As in [3]: M. B. Giles, A-L. Haji-Ali 'Multilevel nested simulation for efficient risk estimation', 2018.
        # Numerically computes the optimal starting level of mlmc
        # M <- Number samples to estimate variables
        # tol_ell0  <- Tolerance to accept given level (should be >= 1)
        # Initial parameters
        ell0 = -1
        done = False

        while done == False:  # Loop until we find (approximately) optimal starting level
            ell0 += 1
            print('l0: ', ell0)
            # Obtain MLMC estimates
            ml0 = self.mlmc_ell(ell0, ell0, M)
            ml1 = self.mlmc_ell(ell0+1, ell0, M)

            # Extract relevant parameters
            V0f = ml0[2]/M - (ml0[1]/M)**2  # Variance at level 0
            W0 = ml0[-1]/M  # Work at level 0
            V1 = ml1[0][1]/M - (ml1[0][0]/M)**2  # Variance at level 1
            W1 = ml1[-2]/M  # Work at level 1
            W1f = ml1[-1]/M  # Work of the fine estimator at level 1
            V1f = ml1[2]/M - (ml1[1]/M)**2  # Variance of the fine estimator at level 1

            # Check optimality within factor given by tol_ell0
            if (np.sqrt(V0f*W0) + np.sqrt(V1*W1)) <= tol_ell0*np.sqrt(V1f*W1f):
                done = True

        print('\nOptimal l0: ', ell0, '\n')
        self.ell0 = ell0


    # [2]: Nathan Collier, Abdul-Lateef Haji-Ali, Fabio Nobile et. al. "A continuation multilevel Monte Carlo algorithm". 2015.
    def cmlmc(self, tols, beta = False, alpha = False, M0 = 10**3, kap0 = 0.1, kap1 = 0.1, p = 0.05, err_split = 0.5, \
        paramrange = 5, tmax = 10**10):
        """
        Performs MLMC computation based on continuation MLMC approach as in [2]:
            tols <- Error tolerances to compute estimator at
            beta, alpha <- multilevel correction variance and bias reduction rate
            M0 <- Number of initial samples at levels ell0, ell0+1, ell0+2
            kap0, kap1 <- Confidence in estimates of proportionality constants
            p <- Desired probability of observing error greater than tol
            err_split <- Proportion of mean square error to be attributed to the bias term
            paramrange <- Number previous levels to compute proportionality constants from
            tmax <- Maximum runtime
        """
        if beta == False or alpha == False:
            raise ValueError('Must declare value of beta and alpha')
        # Initial params
        L = 2  # Start with 3 levels ell = 0,1,2
        ells = np.arange(L+1)
        cost = 0  # Store total cost
        Cp = norm.ppf(1 - p/2)  # Used to scale number of samples per level to ensure the correct error
        lhat = 1  # Initial level for parameter estimation
        t0 = perf_counter()


        # Create data containers
        Mlbar = np.zeros(3)  # Number of samples per level
        suml = np.zeros((2,3))  # Running sum and sum of squares  of terms per level
        costl = np.zeros(3)  # Cost per level
        dMl = (M0*np.ones(3)).astype(int)  # Remaining samples to compute per level
        Vells = np.zeros(3)  # Variances per level

        # Output containers
        P_out = []  # mlmc_estimate at each tol
        cost_out = []  # Cost at each tol
        L_out = []  # L at each tol
        tol_out = []  # Tolerances computed within runtime
        Qw_out = []  # Estimates of bias constant
        Qs_out = []  # Estimates of variance constant

        # Run inital hierarchy and update terms
        for ell in range(L + 1):
            sums = self.mlmc_ell(self.ell0 + ell, self.ell0, dMl[ell])
            Mlbar[ell] += dMl[ell]
            suml[:, ell] += sums[0]
            cost += sums[-2]
            costl[ell] += sums[-2]

        # Useful parameters
        W0 = costl[0]/Mlbar[0]  # Cost at level 0
        print('W0: ', W0)
        w_alpha = lambda ell: W0**(-alpha/self.gamma)*2.**(-ell*alpha)*(2**alpha-1)
        s_beta = lambda ell: W0**(beta/self.gamma)*2**(ell*beta)

        # Estimate proportionality constants and variances as in [2]
        Qw = (np.sum(Mlbar[lhat:]*w_alpha(ells[lhat:])**2\
            *s_beta(ells[lhat:])))**(-1)*np.sum(w_alpha(ells[lhat:])*s_beta(ells[lhat:])*suml[0, lhat:])
        Qs = (np.sum(Mlbar[lhat:]))**(-1)*np.sum(s_beta(ells[lhat:])\
            *(suml[1, lhat:] - 2*Qw*w_alpha(ells[lhat:])*suml[0,lhat:] + Mlbar[lhat:]*Qw**2*w_alpha(ells[lhat:])**2))
        Gam3 = 0.5 + kap1*Qs**(-1)*s_beta(ells[lhat:]) + Mlbar[lhat:]/2
        Gam4 = kap1 + 0.5*(suml[1,lhat:] - suml[0,lhat:]**2/Mlbar[lhat:]) \
            + kap0*Mlbar[lhat:]*(suml[0,lhat:]/Mlbar[lhat:] - Qw*w_alpha(ells[lhat:]))/(2*kap0+Mlbar[lhat:])
        Vells[0] = suml[1,0]/Mlbar[0] - (suml[0,0]/Mlbar[0])**2
        Vells[1:] = np.maximum(Gam4/(Gam3 - 0.5), 0)



        # Loop over all error tolerances
        for TOL in tols:
            # Compute optimal # samples per level
            Ml = np.ceil((Cp/err_split/TOL)**2*np.sqrt(Vells/(costl/Mlbar))\
                *np.sum(np.sqrt(Vells*costl/Mlbar)))
            dMl = np.maximum(Ml - Mlbar,0).astype(int)
            print('\nTOL: ', TOL)
            while np.sum(dMl > 0):  # Loop until convergence criteria met
                # Display useful data
                print('\nL: ', L, '\nQw: ', Qw, 'Qs: ', Qs, '\n%-8s%-20s'%('level','dMl'))
                for ell in range(L+1):
                    print('%-8i%-20i'%(ell,dMl[ell]))

                print('\n')
                print('%-8s%-4s%-20s%-4s%-20s%-4s%-20s%-4s%-20s'\
                    %('level','|', 'mean','|', 'variance','|', 'cost','|','M_ell'))
                print(104*'-')

                # Run mlmc hierarchy
                for ell in range(L + 1):
                    if dMl[ell] > 0:
                        if perf_counter() - t0 > tmax:  # Break if execution time too long
                            break
                        sums = self.mlmc_ell(self.ell0 + ell, self.ell0, dMl[ell])  # Obtain mlmc terms with dMl samples
                        Mlbar[ell] += dMl[ell]  # Update number of samples computed at level ell
                        suml[:, ell] += sums[0]  # Update sum and sum of squared terms at level ell
                        cost += sums[-2]  # Update total cost and cost at level ell
                        costl[ell] += sums[-2]
                        print('%-8i%-4s%-20e%-4s%-20e%-4s%-20f%-4s%-20i'\
                            %(ell,'|', suml[0, ell]/Mlbar[ell], '|', suml[1,ell]/Mlbar[ell] - (suml[0,ell]/Mlbar[ell])**2,\
                            '|', costl[ell]/Mlbar[ell],'|', Mlbar[ell]))  # Prints useful data
                if perf_counter() - t0 > tmax:
                    break

                # Estimate proportionality constants as in [2]
                lhat = max(L-paramrange, 1)
                Qw = (np.sum(Mlbar[lhat:]*w_alpha(ells[lhat:])**2\
                    *s_beta(ells[lhat:])))**(-1)*np.sum(w_alpha(ells[lhat:])*s_beta(ells[lhat:])*suml[0, lhat:])
                Qs = (np.sum(Mlbar[lhat:]))**(-1)*np.sum(s_beta(ells[lhat:])\
                    *(suml[1, lhat:] - 2*Qw*w_alpha(ells[lhat:])*suml[0,lhat:] + Mlbar[lhat:]*Qw**2*w_alpha(ells[lhat:])**2))
                Gam3 = 0.5 + kap1*Qs**(-1)*s_beta(ells[1:]) + Mlbar[1:]/2
                Gam4 = kap1 + 0.5*(suml[1,1:] - suml[0,1:]**2/Mlbar[1:]) \
                    + kap0*Mlbar[1:]*(suml[0,1:]/Mlbar[1:] - Qw*w_alpha(ells[1:]))/(2*kap0+Mlbar[1:])
                Vells[0] = suml[1,0]/Mlbar[0] - (suml[0,0]/Mlbar[0])**2
                Vells[1:] = np.maximum(Gam4/(Gam3 - 0.5), 0)

                # (Re-)Compute optimal # samples per level
                Cl = costl/Mlbar
                Ml = np.ceil((Cp/err_split/TOL)**2*np.sqrt(Vells/Cl)\
                    *np.sum(np.sqrt(Vells*Cl)))
                dMl = np.maximum(Ml - Mlbar,0).astype(int)

                if np.sum(dMl>0.01*Mlbar) == 0: # Test for convergence if computed near optimal samples
                    ERR = abs(Qw)*W0**(-alpha/self.gamma)*2**(-alpha*L)\
                        + Cp*np.sqrt(np.sum(Vells/Mlbar))  # Error estimate based on [2]
                    print('ERR: ', ERR)

                    if ERR > TOL:  # If the error is still too large, append a new level
                        L += 1
                        # Append new level to params #####
                        Vells = np.append(Vells, Qs*s_beta(L)**(-1))
                        Mlbar = np.append(Mlbar, 0)
                        suml = np.append(suml, np.array([[0],[0]]), axis = 1)
                        ells = np.append(ells, L)
                        costl = np.append(costl, 0)
                        ##########
                        # Compute new number of samples for each level
                        Cl = np.append(Cl, 2**self.gamma*Cl[-1])
                        Ml = np.ceil((Cp/err_split/TOL)**2*np.sqrt(Vells/Cl)\
                            *np.sum(np.sqrt(Vells*Cl)))
                        dMl = np.maximum(Ml - Mlbar,0)
            if perf_counter() - t0 > tmax:
                break
            P_out.append(np.sum(suml[0,:]/Mlbar))
            cost_out.append(cost)
            L_out.append(L)
            tol_out.append(TOL)
            Qw_out.append(Qw)
            Qs_out.append(Qs)
            print('Estimate: ', P_out[-1], '\n\n'+ 120*'#', '\n')

        # Store data
        self.output.P = P_out
        self.output.cost_mlmc = cost_out
        self.output.L_mlmc = L_out
        self.output.tol_mlmc = tol_out
        self.output.Qw = Qw_out
        self.output.Qs = Qs_out


    def save_mlmc(self, title):
        # Writes parameters from cmlmc computation to file 'title'
        numTol = len(self.output.tol_mlmc)
        file = open(title, "w")
        file.write('tol P cost L Qw Qs\n')
        for i in range(numTol):
            file.write(str(self.output.tol_mlmc[i]) + ' ' + str(self.output.P[i]) \
            + ' ' + str(self.output.cost_mlmc[i]) + ' ' + str(self.output.L_mlmc[i]) +' '+\
            str(self.output.Qw[i]) + ' ' + str(self.output.Qs[i]) + '\n')
        file.close()

    def save(self, title):
        # Write output data to file 'title'
        file = open(title, "w")
        file.write("level M cost Vf V mf m\n")
        for ell in self.output.levels:
            file.write(str(self.output.levels[ell]) + " " + str(self.output.Ms[ell])\
             + " " + str(self.output.costs[ell]) + " " + str(self.output.Vfs[ell]) + " " + str(self.output.Vs[ell]) + " " + \
             str(self.output.mfs[ell]) + " " +  str(self.output.means[ell]) + "\n")
        file.close()
