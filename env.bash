#
#   cd $HOME
#   svn co http://hfag.phys.ntu.edu.tw:6060/repos/env/trunk/ env
#
#   use 
#      type name 
#  to list a function definition
#    http://www-128.ibm.com/developerworks/library/l-bash-test.html
#
#   use 
#       set 
#           to list all functions
#
#     unset -f name 
#              to remove a function
#
#     typeset -F  
#          lists just the names
#
#  http://www.network-theory.co.uk/docs/bashref/ShellFunctions.html
#
#

 ENV_BASE=env
 export ENV_BASE
 DYW_DBG=0
 BASE_DBG=0
 
 [ -r ~/$ENV_BASE/base/base.bash  ] && . ~/$ENV_BASE/base/base.bash
 [ -r ~/$ENV_BASE/scm/scm.bash  ]   && . ~/$ENV_BASE/scm/scm.bash
 [ -r ~/$ENV_BASE/xml/xml.bash  ]   && . ~/$ENV_BASE/xml/xml.bash
 
 if ([ "$NODE_TAG" != "H" ] && [ "$NODE_TAG" != "U" ]) then
     [ -r ~/$ENV_BASE/dyw/dyw.bash  ]   && . ~/$ENV_BASE/dyw/dyw.bash
 fi
 
 if [ "$NODE_TAG" == "G" ];  then
     [ -r ~/$ENV_BASE/workflow/workflow.bash  ]   && .  ~/$ENV_BASE/workflow/workflow.bash
 fi 	 
 




env-u(){ 
  iwd=$(pwd)
  
  if [ "$NODE_TAG" == "$SOURCE_TAG" ]; then
     echo ============= env-u : no svn update is performed as on source node ================
  else
     cd $HOME/$ENV_BASE 
     
     echo ============= env-u : status before update ================
     svn status -u
     svn update
     echo ============= env-u : status after update ================
     svn status -u
     cd $iwd
     
  fi
  echo ============== env-u :  sourcing the env =============
  [ -r $HOME/$ENV_BASE/env.bash ] && . $HOME/$ENV_BASE/env.bash  
}


env-i(){ [ -r $HOME/$ENV_BASE/env.bash ] && . $HOME/$ENV_BASE/env.bash ; }


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
  cd $HOME/$ENV_BASE
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
  cd $HOME/$ENV_BASE
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




