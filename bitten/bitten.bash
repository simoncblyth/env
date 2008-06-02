
bitten-usage(){

cat << EOU

   \$BITTEN_HOME : $BITTEN_HOME
   \$BITTEN_ENV  : $BITTEN_ENV

   bitten-get   :   svn co/up of trunk is recommended, rather than releases 
      ## on hfag ... svn: SSL is not supported  with /data/usr/local/svn/subversion-1.4.0/bin/svn
      ##  BUT the chekout seems OK

   bitten-install :
      ##  g4pb: 
      ##       Installed /Library/Python/2.5/site-packages/Bitten-0.6dev_r547-py2.5.egg
      ##
      ##  hfag:  use "SUDO= bitten-install" 
      ##       Installed /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/Bitten-0.6dev_r547-py2.5.egg    
      ##
   
   bitten-test :
      ##   g4pb:
      ##           Ran 202 tests in 22.584s  FAILED (errors=13)
      ##           failures from lack of figleaf / clearsilver ...
      ##
      ##    hfag:
      ##          after excluding "report"
      ##         Ran 193 tests in 11.074s  FAILED (errors=64)  
      ##
      ##


    bitten-trac-enable  <name>  :  ini-edit in name tracitory trac.ini
        ## following this do:  sudo apachectl restart 
        ##   ... leads to ... TracError: The Trac Environment needs to be upgraded.
 
    bitten-trac-admin upgrade
       ##   ===> Upgrade done.
       ##
       ## and the web interface has a "Build Status" tab 



    bitten-extras-get  : for nose xml plugin and bitten mods to present results
        ##  see  http://bitten.edgewall.org/ticket/147




    which bitten-slave : $(which bitten-slave)

EOU
}


bitten-env(){
  elocal-
  export BITTEN_FOLD=$HOME/bitten
  export BITTEN_HOME=$BITTEN_FOLD/bitten
 
  
  local env
  case $NODE_TAG in
    H) env=env ;;
    G) env=workflow ;;
    *) env=unknown ;;
  esac
  
  export BITTEN_ENV=$env
}

bitten-get(){
  local iwd=$PWD
  mkdir -p $BITTEN_FOLD
  cd $BITTEN_FOLD
  [ ! -d bitten ] && svn co http://svn.edgewall.org/repos/bitten/trunk bitten
  cd bitten
  [ "$PWD" != "$BITTEN_HOME" ] && echo $msg BITTEN_HOME $BITTEN_HOME inconsistency && return 1 
  
  svn info
  svn up
  cd $iwd
}

bitten-install(){
  cd $BITTEN_HOME
  $SUDO python setup.py install
}

bitten-test(){
   cd $BITTEN_HOME
   $SUDO python setup.py test
}

bitten-trac-enable(){
   local name=${1:-$BITTEN_ENV}
   shift 
   trac-ini-
   trac-ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:bitten.\*:enabled
}

bitten-trac-admin(){
   local name=$BITTEN_ENV
   $SUDO trac-admin $SCM_FOLD/tracs/$name $*
}

bitten-trac-perms(){
 
  #bitten-trac-admin permission add anonymous BUILD_EXEC
  #bitten-trac-admin permission add anonymous BUILD_VIEW
  
  bitten-trac-admin permission add blyth BUILD_ADMIN
  bitten-trac-admin permission add blyth BUILD_VIEW
  bitten-trac-admin permission add blyth BUILD_EXEC
  
  ## TODO: fix up a slave user to do this kind of stuff
  bitten-trac-admin permission add slave BUILD_EXEC

  bitten-trac-admin permission list
     
}

bitten-slave-run(){

   local name=$BITTEN_ENV
   bitten-slave $SCM_URL/tracs/$name/builds

}


bitten-slave-minimal(){

   ## the less smarts the slave needs the better 

   local msg="=== $FUNCNAME :"
   local cfg=$ENV_HOME/bitten/$LOCAL_NODE.cfg

   [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

   local iwd=$PWD
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   cd $tmp
   
   bitten-slave $* -v --dump-reports -f $cfg  $SCM_URL/tracs/env/builds
   
   cd $iwd
}


bitten-fluff(){

    local msg="=== $FUNCNAME: $* "
    local fluff=$ENV_HOME/unittest/demo/fluff.txt
    date >> $fluff
    local cmd="svn ci $fluff -m \"$msg\" "
    echo $cmd
    eval $cmd
}


bitten-slave-remote(){

    local def="--dry-run"
    local arg=${1:-$def}
    
    local iwd=$PWD
    local msg="=== $FUNCNAME :"
    local cfg=$ENV_HOME/bitten/$LOCAL_NODE.cfg
     [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
    cd $tmp
    
    local cmd=$(cat << EOC
     bitten-slave -v $arg  -f $cfg 
         --dump-reports 
          --work-dir=.
         --build-dir=
         --keep-files 
            $SCM_URL/tracs/env/builds
EOC)
    echo $cmd
    eval $cmd

   #
   #  the build-dir is by default created within the work-dir with a name like 
   #   build_${config}_${build}   setting it to "" is a convenience for testing
   #  which MUST go together with "--keep-files" to avoid potentially deleting bits of working copy  
   #


    cd $iwd
}






bitten-extras-get(){

   ## 

   mkdir -p $BITTEN_FOLD
   cd $BITTEN_FOLD

   svn co http://bitten.ufsoft.org/svn/BittenExtraTrac/trunk/  bittentrac
   svn co http://bitten.ufsoft.org/svn/BittenExtraNose/trunk/  nosebitten
}


