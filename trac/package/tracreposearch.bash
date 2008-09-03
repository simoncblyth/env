

tracreposearch-usage(){
   package-usage tracreposearch
   cat << EOU
  
      http://www.trac-hacks.org/wiki/RepoSearchPlugin
  
  
EOU

}

tracreposearch-env(){
   elocal-
   package-
   
   trac- 
    
  local branch
  case $(trac-major) in 
     0.11) branch=0.11     ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export TRACREPOSEARCH_BRANCH=$branch
  #export TRACREPOSEARCH_BRANCH=branches/pyndexter
   
}


tracreposearch-revision(){ echo 4139 ; }
tracreposearch-url(){     echo http://trac-hacks.org/svn/reposearchplugin/$(tracreposearch-branch) ; }
tracreposearch-package(){ echo tracreposearch ; }

tracreposearch-fix(){
  local msg="=== $FUNCNAME :"
  echo $msg ... no fix 
}

tracreposearch-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tracreposearch-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tracreposearch-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tracreposearch-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tracreposearch-get(){       package-get       ${FUNCNAME/-*/} $* ; }

tracreposearch-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tracreposearch-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tracreposearch-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tracreposearch-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tracreposearch-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tracreposearch-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tracreposearch-diff(){      package-diff      ${FUNCNAME/-*/} $* ; }
tracreposearch-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tracreposearch-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

tracreposearch-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }


tracreposearch-conf(){
   local msg="=== $FUNCNAME :"
   trac-configure repo-search:include:\*.py:\*.h:\*.cxx:\*.xml repo-search:exclude:\*.png repo-search:index:$(tracreposearch-indexdir)
}

tracreposearch-permission(){
   trac-admin- permission add authenticated REPO_SEARCH
   trac-admin- permission list
}

tracreposearch-prepare(){

   tracreposearch-enable 
   tracreposearch-conf
   tracreposearch-permission
   tracreposearch-prepindex
}


tracreposearch-indexdir(){
  case $NODE_TAG in 
     G) echo /tmp/env/index ;;
     *) echo /tmp/env/tracreposearch/$TRAC_INSTANCE/index ;;
   esac   
}

tracreposearch-prepindex(){

   local dir=$(tracreposearch-indexdir)
   mkdir -p $dir
   apache-
   local user=$(apache-user)
   $SUDO chown -R  $user:$user $dir

}


tracreposearch-reindex(){

   $SUDO $(tracreposearch-dir)/update-index $(trac-envpath) 
   tracreposearch-prepindex
    
}


tracreposearch-wipeindex(){
  
  local dir=$(tracreposearch-indexdir)
  [ "${dir:0:4}" != "/tmp" ] && echo $msg skipping && return 1 
 
  sudo apachectl stop
  sudo rm -rf $dir/*
  sudo apachectl start   

  tracreposearch-prepindex

}

