
pkg-env(){
   # making minimum usage of environment for transportability
   echo -n
}

pkg-usage(){

   cat << EOU

       \$(which python) : $(which python)

       pkg-importable-  <pkgname>
             detect package by attempting to import ... caution this is 
             dependant on the directory you run it from as well as PYTHONPATH 
             eg :
                 pkg-importable- bitten && echo yep || echo nope


       pkg-eggname-  <pkgname>
             determine the name of the egg from path reported by pkgname.__file__ 
       pkg-eggname <pkgname>
             invokes pkg-eggname- returning a blank in case of any error
           
       pkg-eggrev-  <pkgname>
             determine the svn revision for an egg, if the pkg is not found or does not
             have a revision return a blank
       pkg-eggrev  <pkgname>
             invokes pkg-eggrev- returning a blank in case of any error          
                                
           
                          
       pkg-lastrev <pkgname>
             parses svn info for the working copy at relative path <pkgname>
             plucking the last changed revision reported                                  
                                                        
                                                                                      
       pkg-ezsetup 
             download and run th ez_setup.py script making setuptools available
             to the python in your path, the invoking directory is used as the working
             directory
             
       pkg-install <pkgname> <url> <revision> .. <pkgname2> <url2>  <rev2> ...
             subversion checkout and easy install using the invoking directory as the working directory
             to house svn checkouts/tarballs etc..  
       
       pkg-uninstall <pkgname>
           determine eggname, remove the egg and easy-install.pth reference 
         
           
       pkg-site
            path of python site-packages 
            
       pkg-ls 
            ls of site-packages

                          
      ENHANCEMENT IDEAS:
      
          add support to tgz/zip urls ... not just svn checkouts ?
          
       
                                                 
                          
                                      
EOU

}


pkg-importable-(){  python -c "import $1" 2> /dev/null ; }

pkg-eggname-(){    
   
   # python gives relative paths when done from the source directory containing the package 
   # ... hence the tmp move

python -c "$(cat << EOC
import os ; 
os.chdir('/tmp') ; 
import $1 as _ ; 
eggs=[egg for egg in _.__file__.split('/') if egg.endswith('.egg')] ; 
print eggs[0] 
EOC)"
 
}

pkg-eggname(){
  pkg-eggname- $* 2> /dev/null || echo -n 
}


pkg-eggrev-(){
python -c "$(cat << EOC 
import os ; 
os.chdir('/tmp') ; 
import $1 as _ ; 
eggs=[egg for egg in _.__file__.split('/') if egg.endswith('.egg')] ; 
e=eggs[0] ; 
import re ; 
print re.compile('dev_r(\d*)-').search(e).group(1) 
EOC)"
}

pkg-eggrev(){
   pkg-eggrev- $* 2> /dev/null || echo -n 
}

pkg-eggrev-deprecated-(){
  ## relies on knowing the distribution name ... not the package name ... so better to use other means
python -c "$(cat << EOC
import pkg_resources as pr ; 
d=pr.get_distribution(\"$1\") ; 
v=d.version ; 
p=v.index('dev-r') ; 
print v[p+5:] ; 
EOC)"

}









pkg-uninstall(){

   local msg="=== $FUNCNAME :"
   
   local name=$1
   local eggname=$(pkg-eggname- $name)
   local site=$(pkg-site)
   local eggpath=$site/$eggname
   local pth=$site/easy-install.pth
   local cmd
   
   [ ! -f $pth ] && echo $msg ERROR no pth $pth && return 1
   
   if [ ${#eggpath} -gt 5 -a -e $eggpath ]; then
      
      cmd="$SUDO rm -irf $eggpath "
      echo $cmd
      eval $cmd
      
      cmd="$SUDO env -i perl -pi -e \"s/^\.\/$eggname.*\n$//\" $pth "    
      echo $cmd
      eval $cmd
    
   fi
   
}




pkg-lastrev(){
   ## env -i shields perl from the environment to avoid libutil problem
   #svn info $1 > /dev/stderr
   svn info $1 | env -i perl -n -e 'm/^Last Changed Rev:\s*(\S*)\s*$/ && print $1 ' -
}


pkg-site(){
  python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
}

pkg-ls(){
  ls -l $(pkg-site)
}





pkg-entry-check(){
 for name in $*
  do
     printf "%-20s : %s \n" $name $(which $name)
  done   
}

pkg-info(){
   cat << EOI   
      eggrev  : $(pkg-eggrev $1)
      lastrev : $(pkg-lastrev $1)
EOI
}



pkg-install(){  

   local workdir=$PWD
   pkg-ezsetup
   
   while [ $# -gt 0 ]
   do
     local name=$1
     local url=$2
     local rev=$3
     shift 3
     
     cd $workdir 
     pkg-svnget $name $url $rev 
     
     pkg-install- $name $rev
    
   done
   
   cd $workdir
}



pkg-install-(){

    local msg="=== $FUNCNAME :"
    local name=$1
    local rev=$2
    
    local workdir=$PWD
   
    if [ "$rev" == "HEAD" ]; then
      
      pkg-info $name
      local eggrev=$(pkg-eggrev $name)
      local lastrev=$(pkg-lastrev $name) 
      
      if [ "$eggrev" == "$lastrev" ]; then
         echo $msg the revision of the installed egg $eggrev matches the last changed revision so nothing to do 
      else
         echo $msg the revision of the installed egg $eggrev is not the same as the last changed revision $lastrev, proceed to easy_install
         cd $name
         easy_install . 
      fi
    
    else
        pkg-importable- $name && echo $msg package $name is already importable uninstall it with \"pkg-uninstall $name\" to force reinstallation || easy_install .
    
    fi
    cd $workdir
}





pkg-svnget(){

   local msg="=== $FUNCNAME :"
   
   local iwd=$PWD
   local name=$1
   local url=$2
   local rev=$3
   
   if [ ! -d $name ]; then
        echo $msg initial checkout of $name $url at revision $rev
        svn checkout -r $rev $url $name
   else
        if [ "$rev" == "HEAD" ]; then
            echo $msg update $name revision $rev
            svn update $name
        else
            local ver=$(svnversion $name)
            if [ "$ver" == "$rev" ]; then
                echo $msg $name is already at target revision $ver
            else
                echo $msg updating $name to target revision $rev
                svn update -r $rev  $name
            fi    
        fi
  fi   
  cd $iwd
}



pkg-ezsetup(){

    local msg="=== $FUNCNAME :"
    pkg-importable- setuptools && echo $msg setuptools is already present && return 0 
     
    local ezpy=ez_setup.py
    [ ! -f $ezpy ] && pkg-download http://peak.telecommunity.com/dist/$ezpy
  
    python $ezpy     
  
}


pkg-download () {
    if [ -x "`which wget 2> /dev/null `" ] ; then
        echo "Downloading: wget $@"
        wget -nv $@
    elif [ -x "`which curl 2> /dev/null `" ] ; then
        echo "Downloading: curl $@"
        ## SCB : need -L to follow HTTP 302, temporarily moved ...
        curl -L -O $@
    else
        echo
        echo "No download tool (wget/curl) available."
        exit 1
    fi
}






