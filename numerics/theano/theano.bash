theano-source(){   echo ${BASH_SOURCE} ; }
theano-vi(){       vi $(theano-source) ; }
theano-env(){      elocal- ; }
theano-usage(){ cat << EOU

Theano
=========

Theano is a Python library that allows you to define, optimize, and evaluate 
mathematical expressions involving multi-dimensional arrays efficiently. 

Theano features:

* tight integration with NumPy – Use numpy.ndarray in Theano-compiled functions.
* transparent use of a GPU – Perform data-intensive computations much faster than on a CPU.
* efficient symbolic differentiation – Theano does your derivatives for functions with one or many inputs.
* speed and stability optimizations – Get the right answer for log(1+x) even when x is really tiny.
* dynamic C code generation – Evaluate expressions faster.
* extensive unit-testing and self-verification – Detect and diagnose many types of errors.


* http://deeplearning.net/software/theano/index.html
* http://deeplearning.net/software/theano/tutorial/index.html


Why Looked into Theano
-----------------------

* Used by PyMC3


EOU
}
theano-dir(){ echo $(local-base)/env/numerics/theano ; }
theano-cd(){  cd $(theano-dir); }
theano-mate(){ mate $(theano-dir) ; }
theano-get(){
   local dir=$(dirname $(theano-dir)) &&  mkdir -p $dir && cd $dir

}
