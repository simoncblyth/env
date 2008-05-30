
bitten-usage(){

cat << EOU

   \$BITTEN_HOME : $BITTEN_HOME
   \$BITTEN_ENV  : $BITTEN_ENV

   bitten-get   :   svn co/up of trunk is recommended, rather than releases 


   bitten-install :
      ## Installed /Library/Python/2.5/site-packages/Bitten-0.6dev_r547-py2.5.egg

    
   bitten-test :
     ##  Ran 202 tests in 22.584s  FAILED (errors=13)
     ##  failures from lack of figleaf / clearsilver ...


    bitten-trac-enable  <name>  :  ini-edit in name tracitory trac.ini
        ## following this do:  sudo apachectl restart 
        ##   ... leads to ... TracError: The Trac Environment needs to be upgraded.
 
    bitten-trac-admin upgrade
       ##   ===> Upgrade done.
       ##
       ## and the web interface has a "Build Status" tab 


    which bitten-slave : $(which bitten-slave)

EOU
}


bitten-env(){
  elocal-
  export BITTEN_HOME=$HOME/bitten
  export BITTEN_ENV=workflow
}


bitten-get(){
  local iwd=$PWD
  mkdir -p $(dirname $BITTEN_HOME)
  cd $(dirname $BITTEN_HOME)
  [ ! -d bitten ] && svn co http://svn.edgewall.org/repos/bitten/trunk bitten
  cd bitten
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
 
  bitten-trac-admin permission add anonymous BUILD_EXEC
  bitten-trac-admin permission add anonymous BUILD_VIEW
  bitten-trac-admin permission add blyth BUILD_ADMIN

  bitten-trac-admin permission list
     
}

