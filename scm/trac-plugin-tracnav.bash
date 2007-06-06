


trac-plugin-tracnav-get(){

   #  documented at 
   # http://svn.ipd.uka.de/trac/javaparty/wiki/TracNav
   #

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins
    
   svn co http://svn.ipd.uka.de/repos/javaparty/JP/trac/plugins/tracnav/
   cd tracnav

}

trac-plugin-tracnav-install(){

    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd tracnav
    python setup.py install 
   
  ##  Installed /data/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracNav-3.92-py2.5.egg
}
