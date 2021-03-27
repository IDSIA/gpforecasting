import GPy
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections.abc import Iterable
import logging
import collections


class GP:

    def __init__(self, frequency, period = 1, Q = 2, priors = True, restarts = 1, normalize = False, loglevel = 0):
        self.logger = logging.getLogger("forgp")    
        self.logger.setLevel(loglevel)  
        self.set_frequency(frequency)
        self.set_period(period)
        
        self.Q = Q
        self.restarts = restarts
        self.normalize = normalize
        
        self.has_priors = priors is not False
        if self.has_priors:
            if priors is True:
                self.logger.info("using default priors")
                priors = self.default_priors()
            elif isinstance(priors, (list,pd.core.series.Series,np.ndarray)):
                self.priors_array(priors)
            else: 
                self.init_priors(priors)

        
        self.logger.info(f"GP priors {priors} {self.has_priors}, Q={self.Q}, restarts={self.restarts}")


    def standard_prior(self, data):
        """ Get the priors hash from the array of priors data 
        
        This method will name the values in the data according to the ordering used in the
        hierarchical probabilistic programming prior estimation code. 
        This is:
        - std devs
        - means (variance, periodic then exp, cos for the different Qs)
        - alpha 
        - beta
        """

        names = ["p_std_var", "p_std_other", "p_mu_var", "p_mu_periodic"]
        if self.Q >= 1:
            names += ["p_mu_exp1", "p_mu_cos1"]
        if self.Q >= 2:
            names += [ "p_mu_exp2", "p_mu_cos2"]
        names += [ "p_alpha", "p_beta" ]

        return dict(zip(names, data))

    def priors_array(self, data):
        """ Set priors from array
        """
        priors = {
            "p_std_var": data[0], "p_std_other": data[1], 
            "p_mu_var": data[2], "p_mu_rbf": data[3], 
            "p_mu_periodic": data[4]
        }

        for i in range(1, self.Q+1):
            priors[f"p_mu_exp{i}"] = data[5 + (i-1)*2]
            priors[f"p_mu_cos{i}"] = data[6 + (i-1)*2]

        self.has_priors = True
        self.init_priors(priors)
    
    def default_priors(self):
        """ Get default prior values """
        
    
        priors = {
                "p_std_var": 1.0, "p_std_other": 1.0, 
                "p_mu_var": -1.5, "p_mu_rbf": 1.1, 
                "p_mu_periodic": 0.2, "p_mu_exp1": -0.7, "p_mu_cos1": 0.5, "p_mu_exp2": 1.1, "p_mu_cos2": 1.6, 
        
            }
  
        return priors

    def init_priors(self, priors):
        """ Initialize the prior parameters crearing the GPy priors """
        self.prior_var = GPy.priors.LogGaussian(priors["p_mu_var"], priors["p_std_var"])
        self.prior_lscal_rbf = GPy.priors.LogGaussian(priors["p_mu_rbf"], priors["p_std_other"]) 
        self.prior_lscal_std_periodic = GPy.priors.LogGaussian(priors["p_mu_periodic"], priors["p_std_other"]) 
                
        if self.Q >= 1: 
            self.prior_lscal_exp_short = GPy.priors.LogGaussian(priors["p_mu_exp1"], priors["p_std_other"])
            self.prior_lscal_cos_short = GPy.priors.LogGaussian(priors["p_mu_cos1"], priors["p_std_other"])
        
        if self.Q == 2:
            self.prior_lscal_exp_long = GPy.priors.LogGaussian(priors["p_mu_exp2"], priors["p_std_other"])
            self.prior_lscal_cos_long = GPy.priors.LogGaussian(priors["p_mu_cos2"], priors["p_std_other"])


    def set_period(self, period = 1):
        """ Set the period of the series

        Multiple expected periods can be supported by providing an array
        """

        # check for non iterables and make an array
        if not isinstance(period, Iterable):
            self.periods = [ period ]
        else:
            self.periods = period

    def set_frequency(self, frequency):
        """ Set the data'sfrequency 
        
        The frequency can be either a standard value (monthly, quarterly, yearly, weekly, daily)
        or a float defining the "resolution" of the timeseries

        Parameters
        ----------
            frequency : str|number
                This can be either a standard value among: monthly, quarterly, yearly and weekly 
                or a numeric value
        """
        if type(frequency) != str:
            self.sampling_freq = frequency
        elif frequency == 'monthly':
            self.sampling_freq = 12
        elif frequency == 'quarterly':
            self.sampling_freq = 4
        elif frequency == 'yearly':
            self.sampling_freq = 1
        elif frequency == 'weekly':
            self.sampling_freq = 365.25/7.0
        else:
            raise Exception(f"wrong frequency: {frequency}")

    def set_q(self, Q):
        """ Set the number of spectral kernels (exp+cos) """

        self.Q = Q

    def do_normalize(self, Y, train = True):
        if train: 
            self.mean = np.mean(Y)
            self.std = np.std(Y, ddof=1)
        return (Y - self.mean) / self.std

    def do_denormalize(self, Y):
        return Y * self.std + self.mean

    def build_gp(self, Yin, X = None):
        """ Fit a gaussian process using the specified train values """
        use_bias = True

        if X is None:
            X = np.linspace(1/self.sampling_freq,len(Yin)/self.sampling_freq,len(Yin))
            X = X.reshape(len(X),1)
        
        Y = self.do_normalize(Yin, train = True) if self.normalize else Yin
        self.Xtrain = X

        #the yearly case is managed on its own.
        lin = GPy.kern.Linear(input_dim=1)
        
        if self.has_priors:
            self.logger.debug(f"Setting Variance Prior {self.prior_var}")
            lin.variances.set_prior(self.prior_var)
        K = lin

        if use_bias: 
            bias = GPy.kern.Bias(input_dim=1)
            if self.has_priors:
                self.logger.debug(f"Setting Bias Prior {self.prior_var}")
                bias.variance.set_prior(self.prior_var)
            K = K + bias
    
        rbf = GPy.kern.RBF(input_dim=1)
        if self.has_priors:
            self.logger.debug(f"Setting RBF priors var {self.prior_var} and lengthscale {self.prior_lscal_rbf}")
            rbf.variance.set_prior(self.prior_var)
            rbf.lengthscale.set_prior(self.prior_lscal_rbf)
            
        K = K + rbf

        for period in self.periods:
            #the second component  is the stdPeriodic
            periodic = GPy.kern.StdPeriodic(input_dim=1)
            periodic.period.fix(period) # period is set to 1 year by default

            if self.has_priors:
                self.logger.debug(f"Setting periodic {period} lscale {self.prior_lscal_std_periodic}")
                periodic.lengthscale.set_prior(self.prior_lscal_std_periodic)
                periodic.variance.set_prior(self.prior_var)
            K = K + periodic


        #now initiliazes the  (Q-1) SM components. Each component is rfb*cos, where
        #the variance of the cos is set to 1.
        for ii in range(0, self.Q):
            cos =  GPy.kern.Cosine(input_dim=1)
            cos.variance.fix(1)
            rbf =  GPy.kern.RBF(input_dim=1) #input dim, variance, lenghtscale
    
            if self.has_priors:
                if (ii==0):
                        rbf.variance.set_prior(self.prior_var)
                        rbf.lengthscale.set_prior(self.prior_lscal_exp_long)
                        cos.lengthscale.set_prior(self.prior_lscal_cos_long)
                elif (ii==1):
                        rbf.variance.set_prior(self.prior_var)
                        rbf.lengthscale.set_prior(self.prior_lscal_exp_short)               
                        cos.lengthscale.set_prior(self.prior_lscal_cos_short)
            K = K + cos * rbf
                
        
        GPmodel = GPy.models.GPRegression(X, Y, K)

        if self.has_priors:
            GPmodel.likelihood.variance.set_prior(self.prior_var)
        
    
        try:
            GPmodel.optimize_restarts(self.restarts, robust=True)
        except:
            #in the rare case the single optimization numerically fails
            GPmodel.optimize_restarts(5, robust=True)
        
        self.gp_model = GPmodel
        return GPmodel


    def forecast(self, X_forecast):
        if type(X_forecast) == int:
            lastTrain = self.Xtrain[-1]
            endTest = lastTrain + 1/self.sampling_freq * X_forecast
            X = np.linspace(lastTrain + 1/self.sampling_freq, endTest, X_forecast)
            X = X.reshape(len(X), 1)
        else:
            X = X_forecast
    
        
        m,v = self.gp_model.predict(X)
        s = np.sqrt(v)

        upper = m + s * stats.norm.ppf(0.975)
        
        if self.normalize:
            return self.do_denormalize(m), self.do_denormalize(upper)
        else:
            return m, upper

