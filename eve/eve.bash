eve-usage(){
  cat << EOU

     eve-tethered 
           tether the CMT environment to the NuWa installation

     eve-cmtpp <path>
           prepend the argument to the CMTPROJECTPATH to form a tethered project 
           ... invoked by eve-tethered


     eve-standalone
          try a standanlone config ... bringing in the needed preqs ... 
              root / reflex / gccxml 


EOU

}

eve-info(){

  cat << EOI

      CMTPROJECTPATH : $CMTPROJECTPATH
      ROOTSYS        : $ROOTSYS

      

EOI

}



eve-env(){
  elocal-
}

eve-tethered-deprecated(){

  local msg="=== $FUNCNAME :"
  dyb__
  [ -z "$CMTPROJECTPATH" ] && echo $msg ERROR dyb cmt environment not functional && return 1
  env-cmtpp $ENV_HOME/eve

}



eve-dir(){ echo $ENV_HOME/eve ; }
eve-cd(){  cd $(eve-dir) ; }

eve-split(){
   cd $(eve-dir)/SplitGLView/cmt
   cmt config 
   . setup.sh


}



