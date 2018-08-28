# === func-gen- : numerics/mcmc/mcmc fgp numerics/mcmc/mcmc.bash fgn mcmc fgh numerics/mcmc
mcmc-src(){      echo numerics/mcmc/mcmc.bash ; }
mcmc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mcmc-src)} ; }
mcmc-vi(){       vi $(mcmc-source) ; }
mcmc-env(){      elocal- ; }
mcmc-usage(){ cat << EOU

MCMC : Markov Chain Monte Carlo
==================================

* https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/


See Also
----------

* sxmc-

About MCMC
-------------

* http://www.mcmchandbook.net/HandbookChapter1.pdf

* https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm

* http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/

  Intuition behind MCMC sampling 

* http://docs.pymc.io

  PyMC3 is a Python package for Bayesian statistical modeling and Probabilistic
  Machine Learning focusing on advanced Markov chain Monte Carlo (MCMC) and
  variational inference (VI) algorithms. Its flexibility and extensibility make
  it applicable to a large suite of problems. 



* http://deeplearning.net/software/theano/index.html
* http://deeplearning.net/software/theano/tutorial/index.html


Take a step back and look at MC
-----------------------------------

* https://informs-sim.org/wsc08papers/012.pdf

Fitting using finite MC samples (Barlow and Beeston)

* http://atlas.physics.arizona.edu/~kjohns/teaching/phys586/s06/barlow.pdf



EOU
}
mcmc-dir(){ echo $(local-base)/env/numerics/mcmc/numerics/mcmc-mcmc ; }
mcmc-cd(){  cd $(mcmc-dir); }
mcmc-mate(){ mate $(mcmc-dir) ; }
mcmc-get(){
   local dir=$(dirname $(mcmc-dir)) &&  mkdir -p $dir && cd $dir

}
