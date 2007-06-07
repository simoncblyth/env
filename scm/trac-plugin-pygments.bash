

trac-pygments-plugin-get(){

## http://trac-hacks.org/wiki/TracPygmentsPluginA
##
##  

   cd $LOCAL_BASE/trac
   mkdir -p plugins && cd plugins
   
   nam=tracpygmentsplugin
   zip=$nam.zip
   test -f $zip || curl -o $zip "http://trac-hacks.org/changeset/latest/tracpygmentsplugin?old_path=/&filename=tracpygmentsplugin&format=zip"

   unzip -l $zip
   test -d $nam || unzip $zip
   
   cd $nam
   cd 0.10
   python setup.py install

# Installed /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/TracPygments-0.3dev-py2.5.egg

}