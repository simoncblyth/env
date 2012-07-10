trac2mediawiki-vi(){ vi $BASH_SOURCE ; }
trac2mediawiki-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    trac2mediawiki-fix :
        inplace edit to allow the top level headings to be listed
        without having to specify the page name in the [[TOC]]    
   
   hmm there is a boat load of wiki-macros too 
   

Investigate the hookup to system python on G::

   /usr/bin/python -c "import sys ; print '\n'.join(sys.path)" 

Done via egg: /Library/Python/2.5/site-packages/Trac2MediaWiki-0.0.2-py2.5.egg





EOU

}

trac2mediawiki-env(){
  elocal-
  package-
  trac-

  #local branch=trunk/0.11
  #case $(trac-major) in 
  #   0.11) branch=trunk/0.11 ;;  ## give it a go
  #      *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  #esac

  export TRAC2MEDIAWIKI_BRANCH=trunk/0.11

}


trac2mediawiki-revision(){
   echo 82   
}

trac2mediawiki-url(){     
   trac-
   echo $(trac-localserver)/repos/tracdev/trac2mediawiki/$(trac2mediawiki-branch) 
}
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



trac2mediawiki-branch(){    package-fn  $FUNCNAME $* ; }
trac2mediawiki-basename(){  package-fn  $FUNCNAME $* ; }
trac2mediawiki-dir(){       package-fn  $FUNCNAME $* ; }  
trac2mediawiki-egg(){       package-fn  $FUNCNAME $* ; }
trac2mediawiki-get(){       package-fn  $FUNCNAME $* ; }    

trac2mediawiki-install(){   package-fn  $FUNCNAME $* ; }
trac2mediawiki-uninstall(){ package-fn  $FUNCNAME $* ; } 
trac2mediawiki-reinstall(){ package-fn  $FUNCNAME $* ; }
trac2mediawiki-enable(){    package-fn  $FUNCNAME $* ; }  

trac2mediawiki-status(){    package-fn  $FUNCNAME $* ; } 
trac2mediawiki-auto(){      package-fn  $FUNCNAME $* ; } 
trac2mediawiki-diff(){      package-fn  $FUNCNAME $* ; } 
trac2mediawiki-rev(){       package-fn  $FUNCNAME $* ; } 
trac2mediawiki-cd(){        package-fn  $FUNCNAME $* ; } 

trac2mediawiki-fullname(){  package-fn  $FUNCNAME $* ; } 
trac2mediawiki-update(){    package-fn  $FUNCNAME $* ; } 







