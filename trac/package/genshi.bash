genshi-usage(){
   package-fn  $FUNCNAME $*
   cat << EOU
   
     not needed for TRAC 0.10 
          ... but it would do no harm
            
     tags/0.4.4  ... does not work with TracTags ListTagged ... so use trunk
     trunk

   
     
EOU

}

genshi-env(){
  elocal-
  package-
  
  local branch
  local tm=$(trac-major)
  case $tm in 
     0.10) branch=SKIP         ;;
     0.11) branch=tags/0.5.0        ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  
  #echo $msg trac-major $tm branch $branch  
  export GENSHI_BRANCH=$branch
  
}

 
genshi-branch2revision(){
   case $1 in 
  ##  tags/0.5.0) echo 873 ;;
      tags/0.5.0) echo 896 ;;
               *) echo HEAD ;;
   esac             
} 
 
genshi-revision(){
   echo $(genshi-branch2revision $(genshi-branch))
} 
 
genshi-url(){     echo http://svn.edgewall.org/repos/genshi/$(genshi-branch) ;}
genshi-package(){ echo genshi ; }

genshi-fix(){
   cd $(genshi-dir)   
   echo no fixes
}



genshi-branch(){    package-fn  $FUNCNAME $* ; }
genshi-basename(){  package-fn  $FUNCNAME $* ; }
genshi-dir(){       package-fn  $FUNCNAME $* ; } 
genshi-egg(){       package-fn  $FUNCNAME $* ; }
genshi-get(){       package-fn  $FUNCNAME $* ; }

genshi-install(){   package-fn  $FUNCNAME $* ; }
genshi-uninstall(){ package-fn  $FUNCNAME $* ; }
genshi-reinstall(){ package-fn  $FUNCNAME $* ; }
genshi-enable(){    package-fn  $FUNCNAME $* ; }

genshi-status(){    package-fn  $FUNCNAME $* ; }
genshi-auto(){      package-fn  $FUNCNAME $* ; }
genshi-diff(){      package-fn  $FUNCNAME $* ; } 
genshi-rev(){       package-fn  $FUNCNAME $* ; } 
genshi-cd(){        package-fn  $FUNCNAME $* ; }

genshi-fullname(){  package-fn  $FUNCNAME $* ; }

