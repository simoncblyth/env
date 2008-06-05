tractoc-usage(){
   plugins-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    tractoc-fix :
        inplace edit to allow the top level headings to be listed
        without having to specify the page name in the [[TOC]]    
   
EOU

}

tractoc-env(){
  elocal-
  tplugins-
  
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


tractoc-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
tractoc-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
tractoc-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
tractoc-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
tractoc-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
tractoc-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
tractoc-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
tractoc-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
tractoc-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
tractoc-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
tractoc-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }

tractoc-status(){    plugins-status    ${FUNCNAME/-*/} $* ; }
tractoc-auto(){      plugins-auto      ${FUNCNAME/-*/} $* ; }
tractoc-diff(){      plugins-diff      ${FUNCNAME/-*/} $* ; } 










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
