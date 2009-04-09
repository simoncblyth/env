fullblog-src(){    echo trac/package/fullblog.bash ; }
fullblog-source(){ echo ${BASH_SOURCE:-$(env-home)/$(fullblog-source)} ; }
fullblog-vi(){     vi  $(fullblog-source) ; }


fullblog-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
  
     http://trac-hacks.org/wiki/FullBlogPlugin 
      http://trac-hacks.org/browser/fullblog

   Setting up an instance to use fullblog...   
          
     1) set the trac config and permissions
      
         trac-
         fullblog-
         SUDO=sudo TRAC_INSTANCE=dybsvn  fullblog-prepare  
     
     2) check status of http://cms01.phys.ntu.edu.tw/tracs/dybsvn
        should inform that upgrade needed
        
     3) perform the upgrade with
          SUDO=sudo TRAC_INSTANCE=dybsvn trac-admin- upgrade 
        
     4) check again  http://cms01.phys.ntu.edu.tw/tracs/dybsvn
        there should be an extra "builds" tab if you are logged in 
        as a user with permission to see it 
      
    
                                
EOU

}


fullblog-notes(){

cat << EON


EON
}



fullblog-env(){
  elocal-
  package-
  export FULLBLOG_BRANCH=0.11
}

fullblog-revision(){  echo 5336 ; }
fullblog-url(){       echo http://trac-hacks.org/svn/fullblogplugin/$(fullblog-branch) ; }
fullblog-package(){   echo tracfullblog ; }

fullblog-fix(){
   local msg="=== $FUNCNAME :"
   cd $(fullblog-dir)   
   echo no fixes
}

fullblog-perms(){

 local msg="=== $FUNCNAME :"
 echo $msg for consistency these are now done in trac/tracperm.bash  
 
 # trac-admin- permission add blyth BUILD_ADMIN
 # trac-admin- permission add authenticated BUILD_VIEW
 # trac-admin- permission add authenticated BUILD_EXEC
 # trac-admin- permission list

}


fullblog-prepare(){
   fullblog-enable $*
   fullblog-perms $*
}

fullblog-makepatch(){  package-fn $FUNCNAME $* ; }
fullblog-applypatch(){ package-fn $FUNCNAME $* ; }

fullblog-branch(){    package-fn $FUNCNAME $* ; }
fullblog-basename(){  package-fn $FUNCNAME $* ; }
fullblog-dir(){       package-fn $FUNCNAME $* ; } 
fullblog-egg(){       package-fn $FUNCNAME $* ; }
fullblog-get(){       package-fn $FUNCNAME $* ; }

fullblog-install(){   package-fn $FUNCNAME $* ; }
fullblog-uninstall(){ package-fn $FUNCNAME $* ; }
fullblog-reinstall(){ package-fn $FUNCNAME $* ; }
fullblog-enable(){    package-fn $FUNCNAME $* ; }

fullblog-status(){    package-fn $FUNCNAME $* ; }
fullblog-auto(){      package-fn $FUNCNAME $* ; }
fullblog-diff(){      package-fn $FUNCNAME $* ; } 
fullblog-rev(){       package-fn $FUNCNAME $* ; } 
fullblog-cd(){        package-fn $FUNCNAME $* ; }

fullblog-fullname(){  package-fn $FUNCNAME $* ; }
fullblog-update(){    package-fn $FUNCNAME $* ; }





