trac2mediawiki-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    trac2mediawiki-fix :
        inplace edit to allow the top level headings to be listed
        without having to specify the page name in the [[TOC]]    
   
   hmm there is a boat load of wiki-macros too 
   
   
EOU

}

trac2mediawiki-env(){
  elocal-
  tpackage-
  
  local branch
  case $(trac-major) in 
     0.10) branch=trunk/0.10 ;;
     0.11) branch=trunk/0.11 ;;  ## give it a go
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export TRAC2MEDIAWIKI_BRANCH=$branch

}


trac2mediawiki-url(){     echo http://dayabay.phys.ntu.edu.tw/repos/tracdev/trac2mediawiki/$(trac2mediawiki-obranch) ;}
trac2mediawiki-package(){ echo trac2mediawiki ; }

trac2mediawiki-reldir(){
  ## relative path to get from the checked out folder to the one containing the setup.py
  ## due to the non-standard layout 
    echo plugins
}

trac2mediawiki-fix(){
   if [ "$(trac-major)" == "0.10" ]; then
     echo WARNING THERE ARE WIKI-MACROS ESSENTIAL FOR THE OPERATION OF THIS PACKAGE ...
   fi
}





trac2mediawiki-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
trac2mediawiki-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
trac2mediawiki-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
trac2mediawiki-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
trac2mediawiki-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
trac2mediawiki-get(){       package-get       ${FUNCNAME/-*/} $* ; }

trac2mediawiki-install(){   package-install   ${FUNCNAME/-*/} $* ; }
trac2mediawiki-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
trac2mediawiki-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
trac2mediawiki-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

trac2mediawiki-status(){    package-status    ${FUNCNAME/-*/} $* ; }
trac2mediawiki-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
trac2mediawiki-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
trac2mediawiki-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
trac2mediawiki-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

trac2mediawiki-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }



trac2mediawiki-place-macros(){

   local instance=${1:-$TRAC_INSTANCE}
   local msg="=== $FUNCNAME :"
   
   [ "$(trac-major)" != "0.10" ] && echo $msg not needed for non 0.10 trac-major && return 1
   
   cd $(package-odir- trac2mediawiki) 
   
   echo $msg ===\> instance $instance ... \( plugins folder not wiki-macros \)
   local ifold=$SCM_FOLD/tracs/$instance
   local cmd="$SUDO -u $APACHE2_USER cp -f wiki-macros/* $ifold/plugins/"
   echo $cmd
   eval $cmd

}


trac2mediawiki-remove-macros(){
   local instance=${1:-$TRAC_INSTANCE}
   local msg="=== $FUNCNAME :"
   
   [ "$(trac-major)" != "0.10" ] && echo $msg not needed for non 0.10 trac-major && return 1
   
   local ifold=$SCM_FOLD/tracs/$instance
   echo $msg  
   local cmd="$SUDO -u $APACHE2_USER rm -f $ifold/plugins/MW* $ifold/plugins/Latex*  "
   
   echo $cmd
   echo $msg enter YES to proceed
   read answer
   
   [ "$answer" != "YES" ] && return 0
  
   eval $cmd
}



trac2mediawiki-prepare(){

    
    trac2mediawiki-place-macros $*
    trac2mediawiki-enable $*

}

