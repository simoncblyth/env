#!/usr/bin/env python 
"""
Needs scipy so use the conda python3

ipython --profile=foo

"""
import numpy as np

u,v = np.meshgrid( np.linspace(0,np.pi,50), np.linspace(0,2*np.pi,50) )

uu = u.ravel()
vv = v.ravel()

N = len(uu)
R = 10 

sph = np.zeros( [N, 3] )
sph[:,0] = np.sin(uu)*np.cos(vv)  
sph[:,1] = np.sin(uu)*np.sin(vv)  
sph[:,2] = np.cos(uu)  
sph *= R 


par0 = np.array( [0,0,R/2, 1] ) 
d = np.sqrt(np.sum((sph - par0[:3])**2, axis=1 ))     ## distances from sph points to p 
t = d + par0[3]*np.random.randn(len(d)) 

closest = sph[np.argmin( np.sum( (sph - par0[:3])**2 , axis=1 ) )] 
print("closest sph point to p ", closest)

plot = False
if plot:
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D( sph[:,0], sph[:,1], sph[:,2] , c=t )
    plt.show()
pass


from scipy.optimize import minimize
import scipy.stats as stats

par = np.array( [0,0,0,1] )

t_model = lambda par:np.sqrt(np.sum((sph - par[:3])**2, axis=1 )) 

NLL = lambda par:-np.sum( stats.norm.logpdf(t, loc=t_model(par), scale=par[3] ))

res = minimize(NLL, par, method='nelder-mead')

print("par0 (truth)", par0)
print("NLL result", res.x)
print("delta", res.x-par0)


dir_ = "/tmp/recon"
import os
os.makedirs(dir_, exist_ok=True )

np.save(os.path.join(dir_, "t.npy"), t )
np.save(os.path.join(dir_, "sph.npy"), sph )


