
swig-env(){

   elocal-   
   export SWIG_NAME=swig-1.3.29
   export SWIG_HOME=$SYSTEM_BASE/swig/$SWIG_NAME
}

swigbuild-(){ . $ENV_HOME/swig/swigbuild/swigbuild.bash && swigbuild-env $* ; }


