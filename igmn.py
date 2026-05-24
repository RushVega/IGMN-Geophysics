import numpy as np
from scipy.stats import chi2

class IGMN:
    def __init__(self, tau_nov, delta, sp_min, v_min):
        self.tau_nov = tau_nov
        self.delta = delta
        self.sp_min = sp_min
        self.v_min = v_min
        self.K = 0
        self.priors = []
        self.sp = []
        self.mu = []
        self.C = []
        self.ages = []
        self.D = None
        self.max_distance = None

    def fit(self, X):
        self.D = X.shape[1]
        
        # MATLAB igmn.m (calculation sigma_ini)
        _min = np.min(X, axis=0)
        _max = np.max(X, axis=0)
        range_diff = _max - _min
        range_diff[range_diff == 0] = 1e-6 
        self.sigma_ini = (self.delta * range_diff) ** 2
        
        self.max_distance = chi2.ppf(1.0 - self.tau_nov, self.D)
        
        for i in range(X.shape[0]):
            self.fit_step(X[i])

    def _add_component(self, x):
        self.mu.append(np.copy(x))
        self.C.append(np.diag(self.sigma_ini))
        self.sp.append(1.0)
        self.ages.append(1)
        self.K += 1
        total_sp = sum(self.sp)
        self.priors = [sp / total_sp for sp in self.sp]

    def _prune_components(self):
        indices_to_keep = []
        for j in range(self.K):
            if self.ages[j] > self.v_min and self.sp[j] < self.sp_min:
                continue
            indices_to_keep.append(j)
            
        if len(indices_to_keep) < self.K:
            self.priors = [self.priors[i] for i in indices_to_keep]
            self.sp = [self.sp[i] for i in indices_to_keep]
            self.mu = [self.mu[i] for i in indices_to_keep]
            self.C = [self.C[i] for i in indices_to_keep]
            self.ages = [self.ages[i] for i in indices_to_keep]
            self.K = len(indices_to_keep)
            total_sp = sum(self.sp)
            if total_sp > 0:
                self.priors = [sp / total_sp for sp in self.sp]

    def fit_step(self, x):
        if self.K == 0:
            self._add_component(x)
            return
            
        d2_list = np.zeros(self.K)
        loglikes = np.zeros(self.K)
        
        for j in range(self.K):
            # regularization (minCov MATLAB) 
            C_reg = self.C[j] + np.eye(self.D) * 1e-8
            
            try:
                C_inv = np.linalg.inv(C_reg)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C_reg)
                
            diff = x - self.mu[j]
            d2 = np.dot(diff, np.dot(C_inv, diff)) # Square of the distance of Mahalanobis
            d2_list[j] = d2
            
            sign, logdet = np.linalg.slogdet(C_reg)
            if sign <= 0: logdet = np.log(1e-100)
            
            loglikes[j] = -0.5 * (d2 + logdet + self.D * np.log(2 * np.pi))
            
        # MATLAB learn.m: if the distance to all neurons is greater than the threshold
        if np.all(d2_list > self.max_distance):
            self._add_component(x)
            return
            
        # Overflow protection in exponent
        log_priors = np.log(np.array(self.priors) + 1e-100)
        log_posteriors_unnorm = loglikes + log_priors
        
        max_log_post = np.max(log_posteriors_unnorm)
        posteriors_unnorm = np.exp(log_posteriors_unnorm - max_log_post)
        posteriors = posteriors_unnorm / np.sum(posteriors_unnorm)
        
        # Upgrade
        for j in range(self.K):
            sp_new = self.sp[j] + posteriors[j]
            w_j = posteriors[j] / sp_new if sp_new > 0 else 0.0 
            
            delta_mu = x - self.mu[j]
            mu_new = self.mu[j] + w_j * delta_mu
            
            shift = mu_new - self.mu[j]
            shift_matrix = np.outer(shift, shift)
            
            diff_new = x - mu_new
            diff_new_matrix = np.outer(diff_new, diff_new)
            
            C_new = self.C[j] - shift_matrix + w_j * (diff_new_matrix - self.C[j])
            C_new = (C_new + C_new.T) / 2.0 
            
            self.sp[j] = sp_new
            self.mu[j] = mu_new
            self.C[j] = C_new
            self.ages[j] += 1
            
        total_sp = sum(self.sp)
        for j in range(self.K):
            self.priors[j] = self.sp[j] / total_sp
            
        self._prune_components()

    def predict_conditional(self, x_in, in_indices, out_indices):
        # MATLAB recall.m
        F = len(out_indices)
        D_in = len(in_indices)
        
        loglikes = np.zeros(self.K)
        xm = np.zeros((self.K, F))
        cond_C = []
        
        for j in range(self.K):
            mu_in = self.mu[j][in_indices]
            mu_out = self.mu[j][out_indices]
            
            C_in_in = self.C[j][np.ix_(in_indices, in_indices)] + np.eye(D_in) * 1e-8
            C_out_in = self.C[j][np.ix_(out_indices, in_indices)]
            C_out_out = self.C[j][np.ix_(out_indices, out_indices)]
            
            try:
                C_in_in_inv = np.linalg.inv(C_in_in)
            except np.linalg.LinAlgError:
                C_in_in_inv = np.linalg.pinv(C_in_in)
            
            diff_in = x_in - mu_in
            
            # Conditional average
            xm[j] = mu_out + np.dot(C_out_in, np.dot(C_in_in_inv, diff_in))
            
            # Условная ковариация
            C_out_given_in = C_out_out - np.dot(C_out_in, np.dot(C_in_in_inv, C_out_in.T))
            C_out_given_in = (C_out_given_in + C_out_given_in.T) / 2.0
            cond_C.append(C_out_given_in)
            
            # Login Plausibility
            d2 = np.dot(diff_in, np.dot(C_in_in_inv, diff_in))
            sign, logdet = np.linalg.slogdet(C_in_in)
            if sign <= 0: logdet = np.log(1e-100)
            loglikes[j] = -0.5 * (d2 + logdet + D_in * np.log(2 * np.pi))

        # Log-sum-exp trick for weights
        log_priors = np.log(np.array(self.priors) + 1e-100)
        log_pajs_unnorm = loglikes + log_priors
        
        max_log_pajs = np.max(log_pajs_unnorm)
        pajs_unnorm = np.exp(log_pajs_unnorm - max_log_pajs)
        pajs = pajs_unnorm / np.sum(pajs_unnorm)
        
        expected_out = np.sum(xm * pajs[:, None], axis=0)
        
        return expected_out, pajs, xm, cond_C