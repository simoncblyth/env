tractrac-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    addendum to tractrac-install
         In addition to installing the package into PYTHON_SITE this 
         also installs the trac-admin and tracd entry points by default 
         to /usr/local/bin/{trac-admin,tracd}  
         
         
    tractrac-branch2revision :
         note the revision moniker on cms01 was formerly incorrectly 
         the head revision at the time of installation 7326, 
         should use the revision at creation of the tag 
         ...  for patch name matching between machines
         NB makes no difference to the actual code, but would prevent patches
         from being found
         
            
         
         
         
EOU

}

tractrac-env(){
  elocal-
  package-
  
  export TRACTRAC_BRANCH=$(tractrac-version2branch $TRAC_VERSION)
}

tractrac-version2branch(){
  case $1 in 
        0.11) echo tags/trac-0.11    ;;
      0.11b1) echo tags/trac-0.11b1  ;;
     0.11rc1) echo tags/trac-0.11rc1 ;; 
      0.10.4) echo tags/trac-0.10.4  ;;
       trunk) echo trunk ;;
  esac
}

tractrac-branch2revision(){
   case $1 in 
      tags/trac-0.11) echo 7236 ;;   
                   *) echo HEAD ;;
   esac
}

tractrac-revision(){
   echo $(tractrac-branch2revision $(tractrac-version2branch $TRAC_VERSION))
}

tractrac-url(){     echo http://svn.edgewall.org/repos/trac/$(tractrac-branch) ;}
tractrac-pkgname(){ echo trac ; }

tractrac-fix(){
   cd $(tractrac-dir)   
   echo no fixes
}

tractrac-update(){   package-fn $FUNCNAME $* ; } 


tractrac-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tractrac-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tractrac-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tractrac-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tractrac-get(){       package-get       ${FUNCNAME/-*/} $* ; }

tractrac-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tractrac-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tractrac-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tractrac-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tractrac-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tractrac-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tractrac-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
tractrac-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tractrac-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

tractrac-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }


