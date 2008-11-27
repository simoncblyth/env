sglv-usage(){
   cat << EOU

      sglv-  
           jump into the CMT environment of $(sglv-dir)

   

EOU
}


sglv-env(){
  elocal-
  env-cmt   
  cmt-
  
  sglv-cd
  cmt--
}

sglv-dir(){ echo $ENV_HOME/eve/SplitGLView ; }
sglv-cd(){  cd $(sglv-dir)/$1 ; }




