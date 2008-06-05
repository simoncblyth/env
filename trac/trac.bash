
tractags-(){          . $ENV_HOME/trac/plugins/tractags.bash  && tractags-env $* ; }
tracnav-(){           . $ENV_HOME/trac/plugins/tracnav.bash   && tracnav-env  $* ; }
tractoc-(){           . $ENV_HOME/trac/plugins/tractoc.bash   && tractoc-env  $* ; }
tractrac-(){          . $ENV_HOME/trac/plugins/tractrac.bash  && tractrac-env $* ; }
genshi-(){            . $ENV_HOME/trac/plugins/genshi.bash    && genshi-env   $* ; }

trac-env(){
   elocal-
   tplugins-
}

trac-names(){
   local iwd=$PWD
   cd $ENV_HOME/trac/plugins   
   for bash in *.bash
   do
      local name=${bash/.bash/}
      if [ "$name" != "plugins" ]; then
         echo $name
      fi
   done
   cd $iwd
}

trac-auto(){
  local msg="=== $FUNCNAME :"
  for name in $(trac-names)
  do
      $name-   ||  (  echo $msg ABORT you must define the precursor $name- in trac/trac.bash && sleep 100000 )
      $name-auto
  done
}

trac-status(){
  local msg="=== $FUNCNAME :"
  for name in $(trac-names)
  do
      $name-  || (  echo $msg ABORT you must define the precursor $name- in trac/trac.bash && sleep 100000 )
      $name-status
  done
}






