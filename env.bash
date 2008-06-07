
#
#  this is not sustainable 
#    ... move to putting top level precursors here only 
#        with the sub precursors defined  in the name/name.bash file
#

env-(){         . $ENV_HOME/env.bash && env-env $* ; }

dyw-(){         . $ENV_HOME/dyw/dyw.bash   && dyw-env   $* ; }
root-(){        . $ENV_HOME/dyw/root.bash  && root-env  $* ; }

dyb-(){         . $ENV_HOME/dyb/dyb.bash  && dyb-env  $* ; }
dybi-(){        . $ENV_HOME/dyb/dybi.bash && dybi-env $* ; }
dybr-(){        . $ENV_HOME/dyb/dybr.bash && dybr-env $* ; }
dybt-(){        . $ENV_HOME/dyb/dybt.bash && dybt-env $* ; }


apache2-(){     . $ENV_HOME/apache/apache2.bash && apache2-env $* ; } 

caen-(){        . $ENV_HOME/caen/caen.bash      && caen-env $* ; } 

base-(){        . $ENV_HOME/base/base.bash    && base-env $* ; } 
local-(){       . $ENV_HOME/base/local.bash   && local-env $* ; }
elocal-(){      . $ENV_HOME/base/local.bash   && local-env $* ; }   ## avoid name clash 
cron-(){        . $ENV_HOME/base/cron.bash    && cron-env $* ; } 

scm-(){         . $ENV_HOME/scm/scm.bash && scm-env $* ; } 
scm-backup-(){  . $ENV_HOME/scm/scm-backup.bash && scm-backup-env $* ; } 

unittest-(){    . $ENV_HOME/unittest/unittest.bash && unittest-env $* ; }
qmtest-(){      . $ENV_HOME/unittest/qmtest.bash  && qmtest-env  $* ; }
nose-(){        . $ENV_HOME/unittest/nose.bash  && nose-env  $* ; }
bitrun-(){      . $ENV_HOME/bitrun/bitrun.bash  && bitrun-env $* ; }

trac-(){        . $ENV_HOME/trac/trac.bash && trac-env $* ; } 
tmacros-(){     . $ENV_HOME/trac/macros/macros.bash  && macros-env $* ; }
tpackage-(){    . $ENV_HOME/trac/package.bash        && package-env $* ; } 

otrac-(){       . $ENV_HOME/otrac/otrac.bash     && otrac-env $* ; } 
trac-conf-(){   . $ENV_HOME/otrac/trac-conf.bash && trac-conf-env $* ; } 
trac-ini-(){    . $ENV_HOME/otrac/trac-ini.bash  && trac-ini-env  $* ; } 
authzpolicy-(){ . $ENV_HOME/otrac/authzpolicy.bash && authzpolicy-env $* ; }

svn-(){         . $ENV_HOME/svn/svn.bash         && svn-env $* ; } 


cvs-(){          . $ENV_HOME/cvs/cvs.bash && cvs-env $* ; } 


sqlite-(){      . $ENV_HOME/db/sqlite.bash && sqlite-env $* ; } 
db-(){          . $ENV_HOME/db/db.bash     && db-env $*     ; }


aberdeen-(){    . $ENV_HOME/aberdeen/aberdeen.bash && aberdeen-env $* ; }

python-(){      . $ENV_HOME/python/python.bash  && python-env $*  ; }
ipython-(){     . $ENV_HOME/python/ipython.bash && ipython-env $* ; }



seed-(){        . $ENV_HOME/seed/seed.bash && seed-env $* ; }
macros-(){      . $ENV_HOME/macros/macros.bash && macros-env $* ; }
offline-(){     . $ENV_HOME/offline/offline.bash && offline-env $* ; }


xml-(){         . $ENV_HOME/xml/xml.bash ; }





  
# the below may not work in non-interactive running ???  
md-(){  local f=${FUNCNAME/-} && local p=$ENV_HOME/$f/$f.bash && [ -r $p ] && . $p ; } 
 
 
env-usage(){
cat << EOU
#
#     type name        list a function definition 
#     set               list all functions
#     unset -f name     to remove a function
#     typeset -F        lists just the names
#
#  http://www.network-theory.co.uk/docs/bashref/ShellFunctions.html
#  http://www-128.ibm.com/developerworks/library/l-bash-test.html
#
#
#   debugging tip .. invoke with bash rather than . when debugging :
# 
#  g4pb:env blyth$ . base/base.bash
#  -bash: [: missing `]'
#  g4pb:env blyth$ bash base/base.bash 
#  base/base.bash: line 100: [: missing `]'
#
#       CAUTION error reporting can be a line off

EOU
}


env-env(){
  local msg="=== $FUNCNAME :"
 
  BASE_DBG=0
  SCM_DBG=0
  XML_DBG=0
  SEED_DBG=0
  DYW_DBG=0
  DYB_DBG=0 
  TZERO_DBG=0   ## the interactive/non-interactive switch use for debugging cron/batch issues 

  # 
  #  a better way to debug [-t 0 ] issues is  
  #       env -i bash -c ' whatever '
  #   * -i prevents the env from being passed along 
  #   * single quotes to protech from "this" shell
  #

  local iwd=$(pwd)
  cd $ENV_HOME
 
  base-  
  scm-    
 
  [ "$NODE_TAG" == "H" -o "$NODE_TAG" == "U" ] && export MSG="$msg skipped dyb- on node $NODE_TAG " || dyb- 
 
  cd $iwd
  alias eu=env-u

}

env-u(){ 
  iwd=$(pwd)
  
  if [ "$NODE_TAG" == "$SOURCE_TAG" ]; then
     echo ============= env-u : no svn update is performed as on source node ================
  else
     cd $ENV_HOME 
     
     echo ============= env-u : status before update ================
     svn status -u
     svn update
     echo ============= env-u : status after update ================
     svn status -u
     cd $iwd
     
  fi
  echo ============== env-u :  sourcing the env =============
  [ -r $ENV_HOME/env.bash ] && . $ENV_HOME/env.bash  
}





env-wiki(){ 

[ 1 == 2 ] && cat << EOD	
   #	
   #  usage examples :	
   #
   #	 env-wiki export WikiStart WikiStart
   #           export the wiki page "WikiStart" to file  "WikiStart"
   #   
   #	 env-wiki import WikiStart WikiStart
   #           import the file into the web app
   #
   #
EOD
	 trac-admin $SCM_FOLD/tracs/env wiki $* ; 
}


env-find(){
  q=${1:-dummy}
  cd $ENV_HOME
  find . -name '*.*' -exec grep -H $q {} \;
}

env-x-pkg(){

  X=${1:-$TARGET_TAG}

  if [ "$X" == "ALL" ]; then
    xs="P H G1"
  else
    xs="$X"
  fi

  for x in $xs
  do	
     base-x-pkg $x
     scm-x-pkg $x
     dyw-x-pkg $x
  done

}


env-x-pkg-not-working(){

  X=${1:-$TARGET_TAG}

  iwd=$(pwd)
  cd $ENV_HOME
  dirs=$(ls -1)
  for d in $dirs
  do
    if [ -d $d ]; then
  		cmd="$d-x-pkg $X"
		echo $cmd
		eval $cmd
 	fi 	
  done

}

env-local-dir(){
   sudo mkdir -p $LOCAL_BASE/env 
   sudo chown $USER $LOCAL_BASE/env
}



env-env

