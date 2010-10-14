svnauthzadmin-vi(){ vi $BASH_SOURCE ; }
svnauthzadmin-usage(){
   package-usage svnauthzadmin
   cat << EOU
  
EOU

}

svnauthzadmin-env(){
   elocal-
   package-
   trac- 
  local branch
  case $(trac-major) in 
     0.11) branch=0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export SVNAUTHZADMIN_BRANCH=$branch
}


svnauthzadmin-upgradeconf(){
   local msg="=== $FUNCNAME :"
   [ "$(trac-major)" != "0.11" ] && echo $msg this is only relevant to 0.11 && return 1
}


#svnauthzadmin-revision(){ echo 3768 ; }
svnauthzadmin-revision(){ echo 9290 ; }

svnauthzadmin-url(){     echo http://trac-hacks.org/svn/svnauthzadminplugin/$(svnauthzadmin-branch) ; }
svnauthzadmin-package(){ echo svnauthzadmin ; }

svnauthzadmin-fix(){
  local msg="=== $FUNCNAME :"
  local dir=$(svnauthzadmin-dir)
  echo $msg ... manual copying is replaced by the auto patching   
}

svnauthzadmin-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
svnauthzadmin-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
svnauthzadmin-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
svnauthzadmin-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
svnauthzadmin-get(){       package-get       ${FUNCNAME/-*/} $* ; }

svnauthzadmin-install(){   package-install   ${FUNCNAME/-*/} $* ; }
svnauthzadmin-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
svnauthzadmin-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
svnauthzadmin-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

svnauthzadmin-status(){    package-status    ${FUNCNAME/-*/} $* ; }
svnauthzadmin-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
svnauthzadmin-diff(){      package-diff      ${FUNCNAME/-*/} $* ; }
svnauthzadmin-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
svnauthzadmin-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

svnauthzadmin-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }
svnauthzadmin-update(){    package-fn $FUNCNAME $* ; }

svnauthzadmin-unconf(){
   local msg="=== $FUNCNAME :"
   local name=${1:-$SCM_TRAC}
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   echo $msg not implemented ... use trac-edit to remove the    components:svnauthzadin.\*:enabled
}

svnauthzadmin-conf(){
   local msg="=== $FUNCNAME :"
   local name=${1:-$SCM_TRAC}
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   trac-ini-
   trac-ini-edit $tini components:svnauthzadmin.\*:enabled
}








