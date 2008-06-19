

dybpy-env(){
  elocal-
}

dybpy-usage(){

cat  << EOU

   which python : $(which python)

  dybpy-setup :   
     invoke setup in develop mode 
     providing a beachhead into NuWa python via an egglink 

         python -c "import dybpy as dp ; dp.syspath()  " 

EOU

}


dybpy-setup(){

  local iwd=$PWD 
  cd $ENV_HOME/dybpy
  python setup.py develop
  cd $iwd

}


dybpy-envtest(){

   local msg="=== $FUNCNAME : "
   local paths="dybgaudi/DybRelease dybgaudi/Simulation/GenTools "
   for p in $paths
   do
       echo $msg $p
       dyb__ $p  > /dev/null
       python -c "import dybpy as dp ; dp.syspath()  "
   done

}