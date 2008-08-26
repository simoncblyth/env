

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
   local cmd="python $ENV_HOME/trac/autocomp/autocomponent.py $(trac-envpath $*) $(trac-administrator)"
   echo $cmd
   eval $cmd
   
   #  on cms01 when using direct $SUDO approacg run into
   #    /data/env/system/python/Python-2.5.1/bin/python: 
   #       error while loading shared libraries: libpython2.5.so.1.0: cannot open shared object file: No such file or directory
   #
   
}


autocomp-sudosync(){
   $SUDO bash -lc "trac- ; autocomp- ; autocomp-sync "
}



autocomp-help(){

   local iwd=$PWD
   cd $ENV_HOME/trac/autocomp
   pydoc autocomponent
   
   cd $iwd
}