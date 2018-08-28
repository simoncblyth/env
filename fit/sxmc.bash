sxmc-source(){   echo ${BASH_SOURCE} ; }
sxmc-vi(){       vi $(sxmc-source) ; }
sxmc-env(){      elocal- ; }
sxmc-usage(){ cat << EOU

SXMC : Signal fitting with a GPU-accelerated Markov Chain Monte Carlo
=======================================================================


EOU
}
sxmc-dir(){ echo $(local-base)/env/fit/sxmc ; }
sxmc-cd(){  cd $(sxmc-dir); }
sxmc-get(){
   local dir=$(dirname $(sxmc-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d sxmc ] && git clone https://github.com/mastbaum/sxmc


}
