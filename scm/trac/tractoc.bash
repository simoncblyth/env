tractoc-get(){

   cd $LOCAL_BASE/trac
   [ -d "wiki-macros" ] || mkdir -p wiki-macros
   cd wiki-macros

   local macro=tocmacro
   mkdir -p $macro
   #svn co http://trac-hacks.org/svn/$macro/0.9/ $macro
   
   easy_install -Z http://trac-hacks.org/svn/$macro/0.10/

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


tractoc-enable(){

## NB the appropriate string is the python package name ...
##  test with  python -c "import tractoc" 
##
   local name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:tractoc.\*:enabled
}

tractoc-test(){
    python -c "import tractoc" 
}



tractoc-install(){

  local macro=tocmacro
  cd $LOCAL_BASE/trac/wiki-macros/$macro

  python setup.py install
  
  # Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/Toc-1.0-py2.5.egg  

}
