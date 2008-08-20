

autocomp-usage(){

  cat << EOU

     autocomp-sync  <env-name>    defaults to TRAC_INSTANCE 

             sync the components (names and owners) as specified by the owner
             properties on directories in the repository with the component list used for ticket creation 

     autocomp-help           
     
             pydoc of the autocomponent module 
  
  
EOU


}


autocomp-env(){
   trac-
}


autocomp-sync(){
   local cmd="$SUDO python $ENV_HOME/trac/autocomp/autocomponent.py $(trac-envpath $*) $(trac-administrator)"
   echo $cmd
   eval $cmd
}

autocomp-help(){

   local iwd=$PWD
   cd $ENV_HOME/trac/autocomp
   pydoc autocomponent
   
   cd $iwd
}