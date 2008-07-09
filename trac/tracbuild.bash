

tracbuild-usage(){

   cat << EOU

      tracbuild-has <name>     : rc 0 if name in \$TRAC_NAMES 
      tracbuild-upush <name>   : append to \$TRAC_NAMES only if not there already   
       
      tracbuild-names-auto     : 
               determines names from <name>.bash files in $(trac-pkgpath)
               
      tracbuild-names          : 
                package list obtained by appending auto determined names to the base names
                    \$TRAC_NAMES_BASE : $TRAC_NAMES_BASE
                
  
                                            
      Functions that loop over the names of the packages ...
    
      tracbuild-diff
      tracbuild-status
      tracbuild-summary        : download/build/installation/patch summary 
      tracbuild-makepatch
      tracbuild-auto           : invoke the build 

   
EOU


}


tracbuild-env(){
   trac-
}



tracbuild-has(){
   local nam=$1
   for has in $TRAC_NAMES
   do
      [ "$has" == "$nam" ] && return 0
   done
   return 1
}

tracbuild-upush(){
   local nam=$1
   ! tracbuild-has $nam && export TRAC_NAMES="$TRAC_NAMES $nam"
}

tracbuild-names(){
   export TRAC_NAMES=$TRAC_NAMES_BASE
   for name in $(tracbuild-names-auto)
   do
      tracbuild-upush $name
   done
   echo $TRAC_NAMES 
}


tracbuild-names-auto(){
   local iwd=$PWD
   cd $(trac-pkgpath)   
   for bash in *.bash
   do
      local name=${bash/.bash/}
      echo $name
   done
   cd $iwd
}

tracbuild-diff(){      tracbuild-f diff ; }
tracbuild-status(){    tracbuild-f status ; }
tracbuild-summary(){   tracbuild-f summary ;  }
tracbuild-makepatch(){ tracbuild-f makepatch ;  }
tracbuild-auto(){      tracbuild-f auto ;  }


tracbuild-f(){
  local msg="=== $FUNCNAME :"
  local f=$1
  for name in $(tracbuild-names)
  do
      $name-   ||  (  echo $msg ABORT you must define the precursor $name- in trac/trac.bash && sleep 100000 )
      package-$f $name
  done
}






