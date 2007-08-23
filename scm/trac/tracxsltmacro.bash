
tracxsltmacro-get(){

   # http://www.trac-hacks.org/wiki/XsltMacro

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins

   local macro=xsltmacro
   mkdir -p $macro
   svn co http://trac-hacks.org/svn/$macro/0.9/ $macro
 

}


tracxsltmacro-install(){
   
   ## for a reinstallation after local changes to the source distro
   
   local macro=xsltmacro
   cd $LOCAL_BASE/trac/plugins/$macro 
   easy_install -Z .

# Processing .
# Running setup.py -q bdist_egg --dist-dir /usr/local/trac/plugins/xsltmacro/egg-dist-tmp-Ya8Z0B
# zip_safe flag not set; analyzing archive contents...
# Adding xslt 0.6 to easy-install.pth file
#
# Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/xslt-0.6-py2.5.egg
# Processing dependencies for xslt==0.6
# Finished processing dependencies for xslt==0.6

}


tracxsltmacro-test(){

  python -c "import xslt"
#
# Traceback (most recent call last):
#  File "<string>", line 1, in <module>
#  File "xslt/__init__.py", line 2, in <module>
#    from Xslt import *
#  File "xslt/Xslt.py", line 64, in <module>
#    import libxml2
# ImportError: No module named libxml2  
#
# hmm on g4pb do not yet have libxml2 ... despite having lxml
#  
  
}


tracxsltmacro-enable(){

   local name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:tractoc.\*:enabled



}
