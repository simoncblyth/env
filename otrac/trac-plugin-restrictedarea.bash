

trac-plugin-restrictedarea-usage(){  cat << EOU

EOU
}

trac-plugin-restrictedarea-install(){

# http://www.trac-hacks.org/wiki/RestrictedAreaPlugin

   easy_install -Z http://trac-hacks.org/svn/restrictedareaplugin/0.10/

#
# Downloading http://trac-hacks.org/svn/restrictedareaplugin/0.10/
# Doing subversion checkout from http://trac-hacks.org/svn/restrictedareaplugin/0.10/ to /tmp/easy_install-pWMeO8/0.10
# Processing 0.10
# Running setup.py -q bdist_egg --dist-dir /tmp/easy_install-pWMeO8/0.10/egg-dist-tmp-3PlkWK
# zip_safe flag not set; analyzing archive contents...
# Adding TracRestrictedArea 1.0.0 to easy-install.pth file
#
# Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracRestrictedArea-1.0.0-py2.5.egg
# Processing dependencies for TracRestrictedArea==1.0.0
#
#
#[g4pb:/usr/local/python/Python-2.5.1/lib/python2.5/site-packages] blyth$ python-crack-egg TracRestrictedArea-1.0.0-py2.5.egg 
#
#   [g4pb:/usr/local/python/Python-2.5.1/lib/python2.5/site-packages] blyth$ sudo chown -R blyth:blyth TracRestrictedArea-1.0.0-py2.5.egg 
#
#
#
#   getting issues :
#        ZipImportError: bad local file header in /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracRestrictedArea-1.0.0-py2.5.egg
#   so remove the egg that I cracked ..
#     cd $PYTHON_SITE ; rm -rf TracRestrictedArea-1.0.0-py2.5.egg
#   and "easy_install -Z" it again,  the -Z cracks the egg automatically
# 
#
# easy_install -Z http://trac-hacks.org/svn/restrictedareaplugin/0.10/
# Downloading http://trac-hacks.org/svn/restrictedareaplugin/0.10/
# Doing subversion checkout from http://trac-hacks.org/svn/restrictedareaplugin/0.10/ to /tmp/easy_install-UMPn4P/0.10
# Processing 0.10
# Running setup.py -q bdist_egg --dist-dir /tmp/easy_install-UMPn4P/0.10/egg-dist-tmp-bjg13j
# zip_safe flag not set; analyzing archive contents...
# Adding TracRestrictedArea 1.0.0 to easy-install.pth file
#
# Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracRestrictedArea-1.0.0-py2.5.egg
# Processing dependencies for TracRestrictedArea==1.0.0
#
#
#



}


trac-plugin-restrictedarea-conf(){

   local name=${1:-$SCM_TRAC}
   
   local perma="components:restrictedarea.filter:enabled" 
   local permb="components:restrictedarea.filter.restrictedareafilter:enabled"
   local restrict="restrictedarea:paths:/wiki/restricted,/wiki/secret"
   
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini "$perma $permb $restrict" 
   
   #  subsequently should see  RESTRICTED_AREA_ACCESS in the available actions
   #   trac-permission $name list
   #
   #   trac-permission hottest add anonymous WIKI_VIEW
   #   trac-permission hottest add authenticated RESTRICTED_AREA_ACCESS
   #
   #   have to put sensitive pages beneath restricted ["restricted/SecretPage"]
   #
   
   
}
