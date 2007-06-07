

trac-plugin-accountmanager-get-and-install(){

   easy_install -Z http://trac-hacks.org/svn/accountmanagerplugin/0.10


  # Downloading http://trac-hacks.org/svn/accountmanagerplugin/0.10
  # Doing subversion checkout from http://trac-hacks.org/svn/accountmanagerplugin/0.10 to /tmp/easy_install-w9nwDp/0.10
  # Processing 0.10
  # Running setup.py -q bdist_egg --dist-dir /tmp/easy_install-w9nwDp/0.10/egg-dist-tmp-nT_2dV
  # Adding TracAccountManager 0.1.3dev-r2171 to easy-install.pth file 
  #
  # Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracAccountManager-0.1.3dev_r2171-py2.5.egg
  # Processing dependencies for TracAccountManager==0.1.3dev-r2171
  #
  #   in addition had to crack the egg ...  SHOULD BE AVOIDED BY THE "-Z"
  #
  # 
  # cd  /usr/local/python/Python-2.5.1/lib/python2.5/site-packages
  # python-crack-egg TracAccountManager-0.1.3dev_r2171-py2.5.egg
  # sudo chown blyth:blyth  TracAccountManager-0.1.3dev_r2171-py2.5.egg  
  #
  # sudo apachectl restart
  #  

}


trac-plugin-accountmanager-conf(){

   ## caution webadmin is a pre-requisite to accountmanager

   local name=${1:-$SCM_TRAC} 
   
   local userfile=$APACHE2_HOME/$SVN_APACHE2_AUTH
   
   
   local base="components:acct_mgr.api:enabled"
   local comps="components:acct_mgr.admin.accountmanageradminpage:enabled components:acct_mgr.web_ui.accountmodule:enabled"  
   local login="components:trac.web.auth.loginmodule:disabled components:acct_mgr.web_ui.loginmodule:enabled"
  
    ## disable registration as no checks
   local regist="components:acct_mgr.web_ui.registrationmodule:disabled"

   ## password setup
   local htdigest="components:acct_mgr.htfile.htdigeststore:enabled  components:acct_mgr.htfile.htpasswdstore:disabled account-manager:password_store:htdigeststore account-manager:htdigest_realm:svn-realm"
   local htpasswd="components:acct_mgr.htfile.htdigeststore:disabled components:acct_mgr.htfile.htpasswdstore:enabled  account-manager:password_store:htpasswdstore"
   local pass="$htpasswd account-manager:password_file:$userfile"
   
   
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   cp $tini /tmp/trac.ini.orig   
   
   ini-edit $tini "$base $comps $pass $login $regist"

   diff $tini /tmp/trac.ini.orig 

}

