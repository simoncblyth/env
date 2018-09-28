#!/usr/bin/env python
"""
https://stackoverflow.com/questions/7718034/maximum-likelihood-estimate-pseudocode

https://onlinecourses.science.psu.edu/stat504/node/29/

"""
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats

N = 10000
#N = 100

tpar = np.array( [5, 2.4, 4 ] )

x = np.linspace(0, 100, N)
y_obs = tpar[0] + tpar[1]*x + np.random.normal(0, tpar[2], len(x))
y_model = lambda par:par[0]+par[1]*x 

NLL = lambda par:-np.sum( stats.norm.logpdf(y_obs, loc=y_model(par), scale=par[2] ))
#NLL = lambda par:-np.sum( stats.norm(y_model(par),par[2]).logpdf(y_obs))


par = np.array([1, 1, 1])
#par = tpar.copy()
res = minimize(NLL, par, method='nelder-mead')

print(res.x)
print(res.x-tpar)


import matplotlib.pyplot as plt
plt.ion()

plt.scatter( x, y_obs , s=0.1 ) 
plt.plot( x, y_model(res.x) )  



