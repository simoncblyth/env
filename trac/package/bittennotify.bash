bittennotify-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
 
      provides email notification ... 
      developed for trac 0.10 ... ported to 0.11 in the patch 
   
     http://trac.3dbits.de/bittennotify/wiki
   
EOU

}

bittennotify-env(){
  elocal-
  package-
  export BITTENNOTIFY_BRANCH=trunk
}

bittennotify-revision(){  echo 28 ; }
bittennotify-url(){     echo http://svn.3dbits.de/bittennotify/$(bittennotify-branch) ;}


bittennotify-unpatch(){
  local msg="=== $FUNCNAME :"
  echo $msg this is needed for a full \"revert\" because the patch adds a file to wc that is not in the repository 
  
  local path=bittennotify/templates/bitten_notify_email.txt
  [ ! -f "$path" ] && echo $msg no file at $path ... PWD $PWD  && return 1
  rm -rf "$path"   && echo $msg removed file at $path  

}



bittennotify-fix(){
   cd $(bittennotify-dir)   
   echo no fix
}


bittennotify-prepare(){
   bittennotify-enable $*
}



bittennotify-makepatch(){  package-fn $FUNCNAME $* ; }



bittennotify-branch(){    package-fn $FUNCNAME $* ; }
bittennotify-basename(){  package-fn $FUNCNAME $* ; }
bittennotify-dir(){       package-fn $FUNCNAME $* ; } 
bittennotify-egg(){       package-fn $FUNCNAME $* ; }
bittennotify-get(){       package-fn $FUNCNAME $* ; }

bittennotify-install(){   package-fn $FUNCNAME $* ; }
bittennotify-uninstall(){ package-fn $FUNCNAME $* ; }
bittennotify-reinstall(){ package-fn $FUNCNAME $* ; }
bittennotify-enable(){    package-fn $FUNCNAME $* ; }

bittennotify-status(){    package-fn $FUNCNAME $* ; }
bittennotify-auto(){      package-fn $FUNCNAME $* ; }
bittennotify-diff(){      package-fn $FUNCNAME $* ; } 
bittennotify-rev(){       package-fn $FUNCNAME $* ; } 
bittennotify-cd(){        package-fn $FUNCNAME $* ; }

bittennotify-fullname(){  package-fn $FUNCNAME $* ; }
bittennotify-update(){    package-fn $FUNCNAME $* ; }




