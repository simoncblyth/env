

trac-usage(){
cat << EOU

   This attempts to automate Trac installation as far as possible
   documentation of the manual installation of 0.11b1 is at 
   
      http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/LeopardTrac

   Considerations...
      0) all packages including plugins to be installed by a standard procedure
      1) do i need to kickstart the tracitory from a prior backup 



EOU

}


tractags-(){          . $ENV_HOME/trac/plugins/tractags.bash  && tractags-env $* ; }
tracnav-(){           . $ENV_HOME/trac/plugins/tracnav.bash   && tracnav-env  $* ; }
tractoc-(){           . $ENV_HOME/trac/plugins/tractoc.bash   && tractoc-env  $* ; }
accountmanager-(){    . $ENV_HOME/trac/plugins/accountmanager.bash    && accountmanager-env   $* ; }
bitten-(){            . $ENV_HOME/trac/plugins/bitten.bash    && bitten-env   $* ; }


## these are not plugins .. hmm package would be a better name
tractrac-(){          . $ENV_HOME/trac/plugins/tractrac.bash  && tractrac-env $* ; }
genshi-(){            . $ENV_HOME/trac/plugins/genshi.bash    && genshi-env   $* ; }


trac-env(){
   elocal-
   tplugins-
   
   case $NODE_TAG in
     G) export TRAC_INSTANCE=workflow ;;
     H) export TRAC_INSTANCE=env ;;
     *) export TRAC_INSTANCE=    ;;
   esac
}


trac-tail(){
  tail -f $(trac-logpath $*)
}

trac-log(){
  cd $(dirname $(trac-logpath $*))
  ls -l 
}

trac-logpath(){
  local name=${1:-$TRAC_INSTANCE}
  echo $SCM_FOLD/tracs/$name/log/trac.log
}


trac-admin-(){
   local name=$TRAC_INSTANCE
   $SUDO trac-admin $SCM_FOLD/tracs/$name $*
}

trac-configure(){
   local name=$TRAC_INSTANCE
   shift
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini 
   trac-ini-
   trac-ini-edit $tini $*
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






