bitten-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    
   Setting up an instance to use bitten...   
          
     1) set the trac config and permissions
      
         trac-
         bitten-
         SUDO=sudo TRAC_INSTANCE=dybsvn  bitten-prepare  
     
     2) check status of http://cms01.phys.ntu.edu.tw/tracs/dybsvn
        should inform that upgrade needed
        
     3) perform the upgrade with
          SUDO=sudo TRAC_INSTANCE=dybsvn trac-admin- upgrade 
        
     4) check again  http://cms01.phys.ntu.edu.tw/tracs/dybsvn
        there should be an extra "builds" tab if you are logged in 
        as a user with permission to see it 
      
      
      
   addendum to bitten-install
       
      In addition to installing the package into PYTHON_SITE this 
      also installs the bitten-slave entry point by default 
      to /usr/local/bin/bitten-slave
     
                                
EOU

}


bitten-notes(){

cat << EON

   bitten-get   :   svn co/up of trunk is recommended, rather than releases 
      ## on hfag ... svn: SSL is not supported  with /data/usr/local/svn/subversion-1.4.0/bin/svn
      ##  BUT the chekout seems OK

   bitten-install :
      ##  g4pb: 
      ##       Installed /Library/Python/2.5/site-packages/Bitten-0.6dev_r547-py2.5.egg
      ##
      ##  hfag:  use "SUDO= bitten-install" 
      ##       Installed /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/Bitten-0.6dev_r547-py2.5.egg    
      ##
   
   bitten-test :
      ##   g4pb:
      ##           Ran 202 tests in 22.584s  FAILED (errors=13)
      ##           failures from lack of figleaf / clearsilver ...
      ##
      ##    hfag:
      ##          after excluding "report"
      ##         Ran 193 tests in 11.074s  FAILED (errors=64)  
      ##
      ##
EON



}



bitten-env(){
  elocal-
  package-
  
  local branch
  case $(trac-major) in 
     0.10) branch=trunk ;;
     0.11) branch=branches/experimental/trac-0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export BITTEN_BRANCH=$branch

}

#bitten-revision(){  echo 547 ; }
bitten-revision(){  echo 556 ; }
bitten-url(){       echo http://svn.edgewall.org/repos/bitten/$(bitten-branch) ;}
bitten-package(){   echo bitten ; }

bitten-fix(){
   cd $(bitten-dir)   
   echo no fixes
}



bitten-makepatch(){  package-fn $FUNCNAME $* ; }
bitten-applypatch(){ package-fn $FUNCNAME $* ; }


bitten-branch(){    package-fn $FUNCNAME $* ; }
bitten-basename(){  package-fn $FUNCNAME $* ; }
bitten-dir(){       package-fn $FUNCNAME $* ; } 
bitten-egg(){       package-fn $FUNCNAME $* ; }
bitten-get(){       package-fn $FUNCNAME $* ; }

bitten-install(){   package-fn $FUNCNAME $* ; }
bitten-uninstall(){ package-fn $FUNCNAME $* ; }
bitten-reinstall(){ package-fn $FUNCNAME $* ; }
bitten-enable(){    package-fn $FUNCNAME $* ; }

bitten-status(){    package-fn $FUNCNAME $* ; }
bitten-auto(){      package-fn $FUNCNAME $* ; }
bitten-diff(){      package-fn $FUNCNAME $* ; } 
bitten-rev(){       package-fn $FUNCNAME $* ; } 
bitten-cd(){        package-fn $FUNCNAME $* ; }

bitten-fullname(){  package-fn $FUNCNAME $* ; }


bitten-perms(){

 local msg="=== $FUNCNAME :"
 
 ## NB the name of the target instance is hidden in TRAC_INSTANCE 

 echo $msg for consistency these are now done in trac/tracperm.bash  
 
 # trac-admin- permission add blyth BUILD_ADMIN
 # trac-admin- permission add authenticated BUILD_VIEW
 # trac-admin- permission add authenticated BUILD_EXEC
 # trac-admin- permission list

}


bitten-prepare(){

   bitten-enable $*
   bitten-perms $*

}


bitten-extras-get(){

   ## 

  cd /tmp

   svn co http://bitten.ufsoft.org/svn/BittenExtraTrac/trunk/  bittentrac
   svn co http://bitten.ufsoft.org/svn/BittenExtraNose/trunk/  nosebitten
}



