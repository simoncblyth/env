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

bitten-url(){     echo http://svn.edgewall.org/repos/bitten/$(bitten-obranch) ;}
bitten-package(){ echo bitten ; }

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

bitten-install(){   package-install   ${FUNCNAME/-*/} $* ; }
bitten-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
bitten-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
bitten-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

bitten-status(){    package-status    ${FUNCNAME/-*/} $* ; }
bitten-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
bitten-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
bitten-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
bitten-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

bitten-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }


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


bitten-extras-get(){

   ## 

  cd /tmp

   svn co http://bitten.ufsoft.org/svn/BittenExtraTrac/trunk/  bittentrac
   svn co http://bitten.ufsoft.org/svn/BittenExtraNose/trunk/  nosebitten
}



