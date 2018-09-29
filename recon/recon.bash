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

A High Performance Implementation of Likelihood Estimators on GPUs
----------------------------------------------------------------------

* https://www.cs.odu.edu/~zubair/papers/CEF2013CreelZubair.pdf

* https://github.com/chokkan/liblbfgs
* http://users.iems.northwestern.edu/~nocedal/lbfgs.html
* http://www.chokkan.org/software/liblbfgs/

  libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)


EOU
}
recon-get(){
   local dir=$(dirname $(recon-dir)) &&  mkdir -p $dir && cd $dir

}
