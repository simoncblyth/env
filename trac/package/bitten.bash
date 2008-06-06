bitten-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
   
   bitten-setver
       the version in the setup.py file ... this is needed when cust is necessary 
   
     
   addendum to bitten-install
       
      In addition to installing the package into PYTHON_SITE this 
      also installs the bitten-slave entry point by default 
      to /usr/local/bin/bitten-slave
     
             
                 
EOU

}

bitten-env(){
  elocal-
  tpackage-
  
  local branch
  case $(trac-major) in 
     0.10) branch=trunk ;;
     0.11) branch=branches/experimental/trac-0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export BITTEN_BRANCH=$branch

}

bitten-url(){     echo http://svn.edgewall.org/repos/bitten/$(bitten-obranch) ;}
bitten-package(){ echo bitten ; }
bitten-eggbas(){  echo Bitten ; }

bitten-setver(){  echo 0.6 ; }  
bitten-eggver(){
    local ob=$(bitten-obranch)
    case $ob in 
                                 trunk) echo 0.6dev_r547 ;;
       branches/experimental/trac-0.11) echo 0.6dev_r542 ;;
                                     *) echo $ob         ;;
    esac
}

bitten-fix(){
   cd $(bitten-dir)   
   echo no fixes
}


bitten-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
bitten-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
bitten-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
bitten-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
bitten-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
bitten-get(){       package-get       ${FUNCNAME/-*/} $* ; }
bitten-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
bitten-install(){   package-install   ${FUNCNAME/-*/} $* ; }
bitten-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
bitten-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
bitten-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

bitten-status(){    package-status    ${FUNCNAME/-*/} $* ; }
bitten-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
bitten-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
bitten-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
bitten-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }



bitten-perms(){

  ## NB the name of the target instance is hidden in TRAC_INSTANCE 

  trac-admin- permission add blyth BUILD_ADMIN
  trac-admin- permission add blyth BUILD_VIEW
  trac-admin- permission add blyth BUILD_EXEC
  
  ## TODO: fix up a slave user to do this kind of stuff
  trac-admin- permission add slave BUILD_EXEC
  trac-admin- permission list

}


bitten-prepare(){

   bitten-enable $*
   bitten-perms $*

}


