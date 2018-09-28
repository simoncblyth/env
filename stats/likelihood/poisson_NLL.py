#!/usr/bin/env python
"""
http://www.astro.ubc.ca/people/jvw/ASTROSTATS/Answers/Chap2/max%20like%20and%20poisson.pdf


  Poisson, population mean mu 

                    -mu     n_i
                   e      mu 
   prob(n_i) =    --------------          probability of count n_i 
                      n_i!

                 -mu
   prob(0) =   e                          probability of no hit 

   prob( n_i > 0 ) = 1 - prob(0)          probability of some counts 
  


   Likelihood( n_i (i=0..N) | mu ) =  product of above for the diffent n_0 n_1 ... 


        LL =  -N mu + sum( n_i ) log mu + const 

        mu = sum( n_i ) / N    ???
    



* :google:`pandel function likelihood` 

* https://astro.desy.de/neutrino_astronomie/icecube/software/icetray_seminar/e693/infoboxContent721/MuonTrackReco_IceTraySeminar.pdf

  exponential function of time residuals (t_res = t_hit - t_geom)  t_geom is expected time with no-scatter or absorption


* https://arxiv.org/abs/0704.1706





"""



