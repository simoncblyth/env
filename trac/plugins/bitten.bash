bitten-usage(){
   plugins-usage  ${FUNCNAME/-*/}
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
  tplugins-
  
  export BITTEN_BRANCH=branches/experimental/trac-0.11_cust
  #export BITTEN_BRANCH=trunk   ## this is for 0.10.4 ?

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


bitten-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
bitten-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
bitten-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
bitten-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
bitten-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
bitten-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
bitten-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
bitten-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
bitten-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
bitten-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
bitten-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }

bitten-status(){    plugins-status    ${FUNCNAME/-*/} $* ; }
bitten-auto(){      plugins-auto      ${FUNCNAME/-*/} $* ; }
bitten-diff(){      plugins-diff      ${FUNCNAME/-*/} $* ; } 
bitten-rev(){       plugins-rev       ${FUNCNAME/-*/} $* ; } 
bitten-cd(){        plugins-cd        ${FUNCNAME/-*/} $* ; }

bitten-perms(){

  trac-admin- permission add blyth BUILD_ADMIN
  trac-admin- permission add blyth BUILD_VIEW
  trac-admin- permission add blyth BUILD_EXEC
  
  ## TODO: fix up a slave user to do this kind of stuff
  trac-admin- permission add slave BUILD_EXEC
  trac-admin- permission list

}


bitten-prepare(){

   bitten-enable
   bitten-perms

}


