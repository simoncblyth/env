#!/usr/bin/env python
"""
SAMPLING A CUMULATIVE DISTRIBUTION FUNCTION
=============================================

Demonstration of:

#. constructing CDF from PDF, by cumulative sum "integration"
#. sampling the CDF to generate a sample with original PDF.

Refs
-----

* :google:`sample cdf`
* http://en.wikipedia.org/wiki/Inverse_transform_sampling
* http://en.wikipedia.org/wiki/Cumulative_distribution_function
* http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
* http://en.wikipedia.org/wiki/Trapezoidal_rule
* http://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html
* http://stackoverflow.com/questions/3209362/how-to-plot-empirical-cdf-in-matplotlib-in-python
* http://matplotlib.org/users/pyplot_tutorial.html
* http://www.math.vt.edu/people/qlfang/class_home/Lesson2021.pdf


Inverse Transform Sampling
--------------------------

* http://en.wikipedia.org/wiki/Inverse_transform_sampling


The cumulative distribution function (CDF) of a distribution 
maps a number in the domain to a probability between 0 and 1.
Inverting the CDF allows to generate a number in the distribution 
from a random number between 0 and 1.



Reminder PDF and CDF
----------------------

#. PDF probability density function::

   I(a->b) p(x) dx   # fraction with a <= x <= b

#. CDF::

   CDF(x) = I(-inf->x)p(t)dt   probability of having value less than x 

   Lim(x->-inf) CDF(x) -> 0
   Lim(x-> inf) CDF(x) -> 1
   CDF cannot decrease to the right 
   density function is derivative of CDF


Use of CDF sampling in Chroma
-------------------------------

chroma/cuda/random.h::

     33 // Sample from a uniformly-sampled CDF
     34 __device__ float
     35 sample_cdf(curandState *rng, int ncdf, float x0, float delta, float *cdf_y)
     36 {
     // range 0.:1.
     37     float u = curand_uniform(rng);
     38 
     39     int lower = 0;
     40     int upper = ncdf - 1;
     41
     // 
     // find straddling bin
     // 
     42     while(lower < upper-1)
     43     {
     44         int half = (lower + upper) / 2;
     45   
     46         if (u < cdf_y[half])
     47             upper = half;
     48         else
     49             lower = half;
     50     }
     51   
     52     float delta_cdf_y = cdf_y[upper] - cdf_y[lower];
     53
     //
     //  effectively this is an inverted CDF function 
     //    
     54     return x0 + delta*lower + delta*(u-cdf_y[lower])/delta_cdf_y;
     55 }
"""

import logging
log = logging.getLogger(__name__)
import numpy as np


def sample_cdf( u, nbin, x0, delta, cdf_y ):
    """
    Reimplement Chroma sample_cdf for elucidatory purposes

    :param u: number from uniform random distribution over range 0:1
    :param nbin: number of bins
    :param x0: low edge
    :param delta: bin size
    :param cdf: numpy array with cumulative distribution function

    :return:  generated value, expected to be distributed according to original distrib 
    """
    lower, upper = 0, nbin - 1

    #
    # find bin of cdf_y that straddles u   
    #
    #     * cdf_y needs to range from 0:1 
    #       (ie a cumulative distribution normalized to 1 at RHS)
    #
    #
    while lower < upper - 1:
        half = (lower + upper)//2
        y = cdf_y[half]
        if u < y:
            upper = half
        else:
            lower = half
        pass
    pass

    #
    #  left edge + whole bins + partial bin
    # 
    delta_cdf_y = cdf_y[upper] - cdf_y[lower]
    return x0 + delta*lower + delta*(u-cdf_y[lower])/delta_cdf_y
 

def make_cdf(entries):
    """
    """
    cs = np.cumsum(entries)
    cdf = cs/cs[-1]    # normalize putting rightmost at 1. 
    return cdf


def multi_sample( u, edges, cdf ):
    """
    :param u: array of uniform random numbers
    :param edges: histogram edges
    :param cdf: cumulative distribution function 
    """
    N = len(u)
    nbin = len(edges) - 1
    binsize = (edges[-1] - edges[0])/nbin
    log.info("multi_sample N %s nbin %s binsize %s " % (N, nbin, binsize))
    r = np.zeros(N)
    for n in range(N):
        r[n] = sample_cdf( u[n], nbin, edges[0], binsize, cdf )
    pass
    return r


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, suppress=True, threshold=20)

    nbin = 35

    # normal distribution sample
    mu, sigma = 0, 0.1 ; s = np.random.normal(mu, sigma, 100000)  
    ext = max(map(abs,[s.min(),s.max()]))  # symmetrical extent
    bins = np.linspace( -ext, ext, nbin+1 ) 

    # construct cumulative distribution function from histogrammed counts
    # source histogram from which to make cdf
    sh = np.histogram( s, bins, density=True )
    counts, edges = sh ; assert len(counts) == len(edges) - 1 
    assert np.all( edges == bins )
    cdf = make_cdf( counts )

    # sample the cumulative distribution function  
    # in attempt to generate sample that resembles the original one
    # ... will tend to reproduce foibles of the original histo 

    u = np.random.rand(10000)  # uniform rand over [0,1]
    r = multi_sample( u, bins, cdf )

    print "u",u
    print "r",r

    rh = np.histogram( r, nbin, density=True )

    import matplotlib.pyplot as plt
    # somewhat of a bias of r to the low side ? 

    print "red: source histogram  blue:histo of distribution generated via cdf sampling "
    plt.plot(sh[1][0:-1], sh[0], 'r-', rh[1][0:-1], rh[0], 'b-')
    plt.show()
    

