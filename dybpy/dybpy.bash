

dybpy-env(){
  elocal-
}

dybpy-usage(){

cat  << EOU

   which python : $(which python)

  dybpy-setup :   
     invoke setup in develop mode 


EOU

}

dybpy-setup(){

  local iwd=$PWD 
  cd $ENV_HOME/dybpy
  python setup.py develop
  cd $PWD

}