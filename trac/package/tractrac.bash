tractrac-src(){    echo trac/package/tractrac.bash  ; }
tractrac-source(){ echo ${BASH_SOURCE:-$(env-home)/$(tractrac-src)} ; }
tractrac-vi(){     vi $(tractrac-source) ; }

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

  ## http://trac.edgewall.org/browser/tags
  case $1 in 
       trunk) echo trunk ;;
      0.11.1) echo tags/trac-0.11.1  ;;
        0.11) echo tags/trac-0.11    ;;
     0.11rc1) echo tags/trac-0.11rc1 ;; 
      0.11b1) echo tags/trac-0.11b1  ;;
      0.10.4) echo tags/trac-0.10.4  ;;
      0.11.4) echo tags/trac-0.11.4  ;;
  esac
}

tractrac-branch2revision(){
   case $1 in 
      tags/trac-0.11) echo 7236 ;; 
    tags/trac-0.11.1) echo 7451 ;;    
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

tractrac-makepatch(){  package-fn $FUNCNAME $* ; }
tractrac-applypatch(){ package-fn $FUNCNAME $* ; }






tractrac-branch(){    package-fn  $FUNCNAME $* ; }
tractrac-basename(){  package-fn  $FUNCNAME $* ; }
tractrac-dir(){       package-fn  $FUNCNAME $* ; }  
tractrac-egg(){       package-fn  $FUNCNAME $* ; }
tractrac-get(){       package-fn  $FUNCNAME $* ; }    

tractrac-install(){   package-fn  $FUNCNAME $* ; }
tractrac-uninstall(){ package-fn  $FUNCNAME $* ; } 
tractrac-reinstall(){ package-fn  $FUNCNAME $* ; }
tractrac-enable(){    package-fn  $FUNCNAME $* ; }  

tractrac-status(){    package-fn  $FUNCNAME $* ; } 
tractrac-auto(){      package-fn  $FUNCNAME $* ; } 
tractrac-diff(){      package-fn  $FUNCNAME $* ; } 
tractrac-rev(){       package-fn  $FUNCNAME $* ; } 
tractrac-cd(){        package-fn  $FUNCNAME $* ; } 

tractrac-fullname(){  package-fn  $FUNCNAME $* ; } 
tractrac-update(){    package-fn  $FUNCNAME $* ; } 







