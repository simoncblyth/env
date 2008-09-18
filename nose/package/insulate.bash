insulate-usage(){
   package-fn  $FUNCNAME $*
   cat << EOU
   
      http://code.google.com/p/insulatenoseplugin/wiki/Documentation
   
   As hfag is down agin ...
   
   simon:fork blyth$ scp /Users/blyth/env/trac/patch/insulate/insulate-trunk-38.patch P:env/trac/patch/insulate/
Scientific Linux CERN Release 3.0.8 (SL)
insulate-trunk-38.patch        
   
   
     env-rsync nose 
     env-rsync python
   
   
EOU
}

insulate-env(){
  elocal-
  package-  
  export INSULATE_BRANCH=trunk
}

insulate-url(){     echo http://insulatenoseplugin.googlecode.com/svn/$(insulate-branch) ; }
insulate-package(){ echo insulate ; }

insulate-fix(){
   cd $(insulate-dir)   
   echo no fixes
}


insulate-develop(){   package-fn $FUNCNAME $* ; }
insulate-makepatch(){ package-fn $FUNCNAME $* ; }
insulate-patchpath(){ package-fn $FUNCNAME $* ; }

insulate-branch(){    package-fn $FUNCNAME $* ; }
insulate-basename(){  package-fn $FUNCNAME $* ; }
insulate-dir(){       package-fn $FUNCNAME $* ; } 
insulate-egg(){       package-fn $FUNCNAME $* ; }
insulate-get(){       package-fn $FUNCNAME $* ; }

insulate-install(){   package-fn $FUNCNAME $* ; }
insulate-uninstall(){ package-fn $FUNCNAME $* ; }
insulate-reinstall(){ package-fn $FUNCNAME $* ; }
insulate-enable(){    package-fn $FUNCNAME $* ; }

insulate-status(){    package-fn $FUNCNAME $* ; }
insulate-auto(){      package-fn $FUNCNAME $* ; }
insulate-diff(){      package-fn $FUNCNAME $* ; } 
insulate-rev(){       package-fn $FUNCNAME $* ; } 
insulate-cd(){        package-fn $FUNCNAME $* ; }

insulate-fullname(){  package-fn $FUNCNAME $* ; }
insulate-update(){    package-fn $FUNCNAME $* ; }
