bitten-src(){    echo trac/package/bitten.bash ; }
bitten-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(bitten-source)} ; }
bitten-vi(){     vi  $(bitten-source) ; }

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


bitten-notes(){ cat << EON


g4pb:~ blyth$ diff e/trac/patch/bitten/bitten-trac-0.11-561.patch bitten-trac-0.11-561.patch
109,121d108
< Index: bitten/htdocs/bitten.css
< ===================================================================
< --- bitten/htdocs/bitten.css  (revision 561)
< +++ bitten/htdocs/bitten.css  (working copy)
< @@ -60,7 +60,7 @@
<  #content.build #charts { clear: right; float: right; width: 44%; }
<  
<  #content.build #builds { clear: none; margin-top: 2em; table-layout: fixed;
< -  width: 54%;
< +  width: 100%;
<  }
<  #content.build #builds tbody th, #content.build #builds tbody td {
<    background: #fff;
272,280d258
< @@ -189,7 +194,7 @@
<          body = str(xml)
<          log.debug('Sending slave configuration: %s', body)
<          resp = self.request('POST', self.url, body, {
< -            'Content-Length': len(body),
< +            'Content-Length': str(len(body)),
<              'Content-Type': 'application/x-bitten+xml'
<          })
<  
g4pb:~ blyth$ 


EON

}



bitten-env(){
  elocal-
  package-
  local branch
  case $(trac-major) in 
     0.11) branch=branches/experimental/trac-0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export BITTEN_BRANCH=$branch
}

#bitten-revision(){  echo 547 ; }
#bitten-revision(){  echo 556 ; }
bitten-revision(){  echo 561 ; }
bitten-url(){       echo http://svn.edgewall.org/repos/bitten/$(bitten-branch) ;}
bitten-package(){   echo bitten ; }

bitten-fix(){
   cd $(bitten-dir)   
   echo no fixes
}

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
  cd /tmp
   svn co http://bitten.ufsoft.org/svn/BittenExtraTrac/trunk/  bittentrac
   svn co http://bitten.ufsoft.org/svn/BittenExtraNose/trunk/  nosebitten
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
bitten-update(){    package-fn $FUNCNAME $* ; }





