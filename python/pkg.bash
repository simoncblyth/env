
pkg-env(){
   # making minimum usage of environment in order to
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
              
       pkg-ezsetup 
             download and run th ez_setup.py script making setuptools available
             to the python in your path, the invoking directory is used as the working
             directory
             
       pkg-install <pkgname> <url> <revision> .. <pkgname2> <url2>  <rev2> ...
             subversion checkout and easy install   
              
EOU

}


pkg-importable-(){  python -c "import $1" 2> /dev/null ; }

pkg-eggname-(){    

   local tmp=/tmp/$FUNCNAME && mkdir -p $tmp
   
   #
   # the python gives relative paths when done from the source directory containing the package 
   # ... hence the tmp shenanigans
   #
      
   cd $tmp
   local iwd=$PWD
   python -c "import $1 as _ ; eggs=[egg for egg in _.__file__.split('/') if egg.endswith('.egg')] ; print eggs[0] " 
   cd $iwd
}



pkg-install(){  

   pkg-ezsetup

   local workdir=/tmp/env/$FUNCNAME && mkdir -p $workdir
   local iwd=$PWD
   
   while [ $# -gt 0 ]
   do
     local name=$1
     local url=$2
     local rev=$3
     shift 3
     
     cd $workdir 
     pkg-svnget $name $url $rev 
     
     cd $name
     easy_install . 
     
   done
   
   #cd $iwd
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






