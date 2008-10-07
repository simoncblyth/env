
trac2latex-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    documented at 
       http://trac-hacks.org/wiki/Trac2LatexPlugin
       http://code.google.com/p/trac2latex/
       
   	 formerly Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracTrac2Latex-0.0.1-py2.5.egg   
	 python setup.py bdist_egg


  
EOU
}



trac2latex-notes(){

  cat << EON
  
      Annoyingly trac2latex uses a non-standard installation approach 
      ... there is no setup.py no package, just a flat plugins folder with loadsa modules

      Try to fix this, by creating a "plugins" package :
  
simon:0.11 blyth$ svn st
?      setup.py
?      plugins/__init__.py


     Launch this onto sys.path in develop mode :

simon:0.11 blyth$ which python
/usr/bin/python

simon:0.11 blyth$ sudo python setup.py develop
Password:
running develop
running egg_info
creating Trac2Latex.egg-info
writing Trac2Latex.egg-info/PKG-INFO
writing top-level names to Trac2Latex.egg-info/top_level.txt
writing dependency_links to Trac2Latex.egg-info/dependency_links.txt
writing entry points to Trac2Latex.egg-info/entry_points.txt
writing manifest file 'Trac2Latex.egg-info/SOURCES.txt'
writing manifest file 'Trac2Latex.egg-info/SOURCES.txt'
running build_ext
Creating /Library/Python/2.5/site-packages/Trac2Latex.egg-link (link to .)
Adding Trac2Latex 0.0.1 to easy-install.pth file

Installed /usr/local/env/trac/package/trac2latex/0.11
Processing dependencies for Trac2Latex==0.0.1
Finished processing dependencies for Trac2Latex==0.0.1

  
  
EON


}


trac2latex-convert(){

   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local path=$1
   local dir=$(dirname $path)
   local name=$(basename $path)
   local base=$(echo $name | cut -d "." -f 1)
   local ftyp=$(echo $name | cut -d "." -f 2)
   
   
   
   [ "$ftyp" != "txt" ] && echo $msg ABORT unexpected file type $path && return 1
   
   cd $dir
   
   local cmd="python `trac2latex-py` $base.txt > $base.tex "
   echo $msg from $dir perform : "$cmd"
   eval $cmd    
   
   
   
   
}

trac2latex-py(){
   echo `trac2latex-dir`/trac2latex/trac2latex.py
}

trac2latex-test(){
   cd `trac2latex-dir`/trac2latex
   python trac2latex.py WikiStart.txt > WikiStart.tex
}


trac2latex-env(){
  elocal-
  package-
  export TRAC2LATEX_BRANCH=$(trac2latex-version2branch $(trac-major))
}


trac2latex-version2branch(){
  case $1 in 
     0.11) echo trunk ;;  
        *) echo version-not-handled ;;
  esac
}


trac2latex-url(){      echo http://dayabay.phys.ntu.edu.tw/repos/tracdev/trac2latex/$(trac2latex-branch) ; }
trac2latex-package(){  echo trac2latex ; }
trac2latex-fix(){      echo -n ; }

trac2latex-prepare(){
     trac2latex-enable $*
}



trac2latex-branch(){    package-fn  $FUNCNAME $* ; }
trac2latex-basename(){  package-fn  $FUNCNAME $* ; }
trac2latex-dir(){       package-fn  $FUNCNAME $* ; }  
trac2latex-egg(){       package-fn  $FUNCNAME $* ; }
trac2latex-get(){       package-fn  $FUNCNAME $* ; }    

trac2latex-install(){   package-fn  $FUNCNAME $* ; }
trac2latex-uninstall(){ package-fn  $FUNCNAME $* ; } 
trac2latex-reinstall(){ package-fn  $FUNCNAME $* ; }
trac2latex-enable(){    package-fn  $FUNCNAME $* ; }  

trac2latex-status(){    package-fn  $FUNCNAME $* ; } 
trac2latex-auto(){      package-fn  $FUNCNAME $* ; } 
trac2latex-diff(){      package-fn  $FUNCNAME $* ; } 
trac2latex-rev(){       package-fn  $FUNCNAME $* ; } 
trac2latex-cd(){        package-fn  $FUNCNAME $* ; } 

trac2latex-fullname(){  package-fn  $FUNCNAME $* ; } 
trac2latex-update(){    package-fn  $FUNCNAME $* ; } 














