tractoc-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    tractoc-fix :
        inplace edit to allow the top level headings to be listed
        without having to specify the page name in the [[TOC]]    
   
EOU

}

tractoc-env(){
  elocal-
  tpackage-
  
  export TRACTOC_BRANCH=0.11_cust
  #export TRACTOC_BRANCH=0.10  
}

tractoc-url(){     echo http://trac-hacks.org/svn/tocmacro/$(tractoc-obranch) ;}
tractoc-package(){ echo tractoc ; }
tractoc-eggbas(){  echo TracTocMacro ; }

tractoc-eggver(){
    local ob=$(tractoc-obranch)
    case $ob in 
       0.10) echo 1.0       ;;
       0.11) echo 11.0.0.3  ;;
          *) echo $ob       ;;
    esac
}

tractoc-fix(){
   cd $(tractoc-dir)   
   perl -pi -e 's/(min_depth.*)2(\s*# Skip.*)/${1}1${2} fixed by tractoc-fix/' tractoc/macro.py
   svn diff tractoc/macro.py
}


tractoc-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
tractoc-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tractoc-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tractoc-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tractoc-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tractoc-get(){       package-get       ${FUNCNAME/-*/} $* ; }
tractoc-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
tractoc-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tractoc-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tractoc-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tractoc-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tractoc-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tractoc-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tractoc-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 










deprecated-tractoc-get(){

   cd $LOCAL_BASE/trac
   [ -d "wiki-macros" ] || mkdir -p wiki-macros
   cd wiki-macros

   local macro=tocmacro
   mkdir -p $macro
   svn co http://trac-hacks.org/svn/$macro/0.10/ $macro
   
   # this works... but want to make a fix first 
   #easy_install -Z http://trac-hacks.org/svn/$macro/0.10/

#
#Downloading http://trac-hacks.org/svn/tocmacro/0.10/
#Doing subversion checkout from http://trac-hacks.org/svn/tocmacro/0.10/ to /tmp/easy_install-zM9Fjm/0.10
#Processing 0.10
#Running setup.py -q bdist_egg --dist-dir /tmp/easy_install-zM9Fjm/0.10/egg-dist-tmp-FQ6C5W
#zip_safe flag not set; analyzing archive contents...
#Adding TracTocMacro 1.0 to easy-install.pth file
#
#Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracTocMacro-1.0-py2.5.egg
#Processing dependencies for TracTocMacro==1.0
#Finished processing dependencies for TracTocMacro==1.0
#



}


deprecated-tractoc-reinstall(){
   
   ## for a reinstallation after local changes to the source distro
   
   local macro=tocmacro
   cd $LOCAL_BASE/trac/wiki-macros/$macro 
   easy_install -Z .


# Processing .
# Running setup.py -q bdist_egg --dist-dir /usr/local/trac/wiki-macros/tocmacro/egg-dist-tmp-i7ylau
# zip_safe flag not set; analyzing archive contents...
# Adding Toc 1.0 to easy-install.pth file
#
# Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/Toc-1.0-py2.5.egg
# Processing dependencies for Toc==1.0
# Finished processing dependencies for Toc==1.0
#

}



deprecated-tractoc-remove(){

  cd $PYTHON_SITE
  rm -rf TracTocMacro-1.0-py2.5.egg
}






deprecated-tractoc-install(){

  local macro=tocmacro
  cd $LOCAL_BASE/trac/wiki-macros/$macro

  python setup.py install
  
  # Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/Toc-1.0-py2.5.egg  

}
