
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
#
#
# Processing .
# Running setup.py -q bdist_egg --dist-dir /data/usr/local/trac/plugins/xsltmacro/egg-dist-tmp-3SrfA1
# zip_safe flag not set; analyzing archive contents...
# Adding xslt 0.6 to easy-install.pth file
#
# Installed /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/xslt-0.6-py2.5.egg
# Processing dependencies for xslt==0.6
#
#

}


tracxsltmacro-test(){

  python -c "import xslt"

#   fails on g4pb , succeeds on hfag
#
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
#   after using the standard approach to building libxml2 and libxslt and python bindings this succeeds on G4PB   
  
}


tracxsltmacro-enable(){

   local name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:xslt.\*:enabled

}


tracxsltmacro-propagate-env(){
 
   local name=${1:-$SCM_TRAC}
   local ini=$SCM_FOLD/tracs/$name/conf/trac.ini
   
   if [ ! -f $ini ]; then
        echo tracxsltmacro-propagate-env error no such trac.ini file $ini  
        return 1 
   fi
   
   local vars="APACHE_LOCAL_FOLDER APACHE_MODE HFAG_PREFIX"

   for var in $vars
   do 
      eval vval=\$$var
      if [ "X$vval" == "X" ]; then
         echo tracxsltmacro-propagate-env error not defined $var
      else   
         local cmd="ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini xslt:$var:$vval"
         echo $cmd
         eval $cmd 
      fi 
   done
         
}