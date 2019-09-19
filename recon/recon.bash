recon-source(){   echo ${BASH_SOURCE} ; }
recon-edir(){ echo $(dirname $(recon-source)) ; }
recon-ecd(){  cd $(recon-edir); }
recon-dir(){  echo $LOCAL_BASE/env/recon/recon ; }
recon-cd(){   cd $(recon-dir); }
recon-vi(){   vi $(recon-source) ; }
recon-env(){  elocal- ; }
recon-usage(){ cat << EOU

Recon Investigations
=======================

* ~/intro_to_numpy/recon.py 
* ~/intro_to_cuda/recon/ 

Minuit2 
-------------

* see minuit2- 



GPURecTimeLikeAlg 

Hit Counting docDB 2219? "nPE recognition"
--------------------------------------------


Metrics for waveform recon
-----------------------------

* https://en.wikipedia.org/wiki/Wasserstein_metric


Liklihood
------------


Galbiati
~~~~~~~~~~

* https://arxiv.org/abs/physics/0503185

Time and space reconstruction in optical, non-imaging, scintillator-based particle detectors
Cristiano Galbiati, Kevin McCarty





CNN on the sphere
---------------------

* :google:`spherical CNN`
* :google:`HealPIX CNN`
* :google:`CMB CNN`

* https://github.com/daniilidis-group/spherical-cnn


Convolutional Neural Networks on the HEALPix sphere: a pixel-based algorithm and its application to CMB data analysis

* https://arxiv.org/abs/1902.04083


:google:`HEALPix CUDA`
------------------------

* https://github.com/elsner/arkcos
* https://arxiv.org/abs/1104.0672


* https://sourceforge.net/projects/healpix/

  * OOPS : HEALPix is GPLv2


Spherical CNNs, Taco S. Cohen, Mario Geiger, Jonas Koehler, Max Welling
--------------------------------------------------------------------------

* https://arxiv.org/abs/1801.10130

* https://github.com/jonas-koehler/s2cnn

  * site has conda intallation instructions 




Another Spherical CNN
------------------------


* https://github.com/daniilidis-group/spherical-cnn








Searching for non-ML GPU use in HEP examples
----------------------------------------------

High Performance Numerical Computing for High Energy Physics: A New Challenge for Big Data Science
Florin Pop

Comparison of MC and Data processing chains : [starting point for a diagram I need to create]

* Figure 1: General approach of event generation, detection, and reconstruction.



He Miao Abstract for CHEP2019 “A first step of event reconstruction in JUNO”
-------------------------------------------------------------------------------

* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=4562



A vertex reconstruction algorithm in the central detector of JUNO
------------------------------------------------------------------

* Qin Liu, et al., JINST (2018) 13 T09005
* http://inspirehep.net/record/1664447?ln=en


A High Performance Implementation of Likelihood Estimators on GPUs
----------------------------------------------------------------------

* https://www.cs.odu.edu/~zubair/papers/CEF2013CreelZubair.pdf

* https://github.com/chokkan/liblbfgs
* http://users.iems.northwestern.edu/~nocedal/lbfgs.html
* http://www.chokkan.org/software/liblbfgs/

  libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)


Papers
--------

Time and space reconstruction in optical, non-imaging, scintillator-based particle detectors

* http://inspirehep.net/record/679088
* Good intro to history of scintillator-based detectors
* per-PMT likelihood
* taylor expansion of likelihood 




Potential Waveform Analysis Algorithms for JUNO emphasized on timing
Benda Xu

* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=4717
* Vertex resolution of JUNO will be dominated by timing.


Event Reconstruction with Machine Learning Algorithms

* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=4633
* Yury Malyshkin

The primary vertex reconstruction using neural networks

* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=4275
* Dmitry Selivanov

* Maximum likelihood method : no training, computationally heavy
* NN : needs training, inference can be v fast
* CNN : Mollweide projection, 75 hrs CNN training : and its quite low resolution




A new method of energy reconstruction for large spherical liquid scintillator detectors
* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=3365

p3: JUNO is 20 times larger than any present liquid scintillator detector and the
profile of expected nPE is deeply affected by absorption and reemission,
Rayleigh scattering, refraction and the total (internal) reflection.




PHYSTAT-nu talk:Machine Learning methods for JUNO Experiment
* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=3915
* Yu Xu
* 



EOU
}
recon-get(){
   local dir=$(dirname $(recon-dir)) &&  mkdir -p $dir && cd $dir

}
