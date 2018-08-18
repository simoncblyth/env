sxmc-source(){   echo ${BASH_SOURCE} ; }
sxmc-vi(){       vi $(sxmc-source) ; }
sxmc-env(){      elocal- ; }
sxmc-usage(){ cat << EOU

SXMC : Signal fitting with a GPU-accelerated Markov Chain Monte Carlo
=======================================================================

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



EOU
}
sxmc-dir(){ echo $(local-base)/env/fit/sxmc ; }
sxmc-cd(){  cd $(sxmc-dir); }
sxmc-get(){
   local dir=$(dirname $(sxmc-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d sxmc ] && git clone https://github.com/mastbaum/sxmc


}
