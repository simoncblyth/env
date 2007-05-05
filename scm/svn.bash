#
#      svn-x
#      svn-i
#      svn-s

#  ...  apache2 connection ....
#
#      svn-apache2-settings                      add the LoadModule lines to httpd.conf .. for mod_dav_svn mod_authz_svn and mod_python 
#
#      svn-apache2-conf                          write the location into SVN_APACHE2_CONF and plant the Include to it in httpd.conf
#        svn-apache2-location                    location "element" for httpd.conf to stdout
#        svn-apache2-authzaccess
#
#      svn-apache2-add-user name                 add a user to the htpasswd file , or change an existing users password
#      svn-apache2-open
#
#    svn-apache2-location-simpleauth-deprecated
#
#    ---------------
#
#     "svn status -u" from the working copy path of interest     
#           should show a single line :  "Status against revision:      8"
# 
#     test that the working copy is a clean revision with smth like:
#
#
# 
#
#    oops preferred version for trac is 1.2.3   ????  http://trac.edgewall.org/wiki/TracSubversion
#


#
# runtime settings including defining and adding SVN_HOME to the PATH are in scm_use.bash
#



export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH


svn-x(){ scp $SCM_HOME/svn.bash ${1:-$TARGET_TAG}:$SCM_BASE; }

svn-i(){ . $SCM_HOME/svn.bash ; }

svn-info(){
  cat << EOI
  LOCAL_BASE $LOCAL_BASE
  SVN_NAME $SVN_NAME
  SVN_ABBREV $SVN_ABBREV
EOI
}


svn-apache2-settings(){
  echo ======= the below LoadModule directives are needed for svn 
  cat << EOS
LoadModule dav_svn_module     libexec/mod_dav_svn.so
LoadModule authz_svn_module   libexec/mod_authz_svn.so
LoadModule wsgi_module        libexec/mod_wsgi.so
EOS

  apache2-add-module dav_svn 
  apache2-add-module authz_svn 
 #apache2-add-module wsgi
  apache2-add-module python

  echo $SUDO vi $APACHE2_HOME/etc/apache2/httpd.conf 
}


svn-apache2-authzaccess-conf(){

  authzaccess=$APACHE2_HOME/$SVN_APACHE2_AUTHZACCESS
  echo =============== writing svn-apache2-authzaccess output to $authzaccess
  $ASUDO svn-apache2-authzaccess >  $authzaccess
  echo =============== cat $authzaccess
  cat $authzaccess

}


svn-apache2-conf(){


  conf=$APACHE2_HOME/$SVN_APACHE2_CONF
  echo =============== writing svn-apache2-location output to $conf
  $ASUDO svn-apache2-location  >  $conf
  echo =============== cat $conf 
  cat $conf 

  svn-apache2-authzaccess-conf 


  xslt=$APACHE2_HTDOCS/resources/xslt
  echo ============== placing stylesheets for raw SVN presentation into $xslt
  mkdir -p $xslt 
  cp -f $SVN_BUILD/tools/xslt/svnindex.* $xslt/ 

  ## correct a braindead absolute path 
  perl -pi -e 's|/svnindex.css|/resources/xslt/svnindex.css|' $xslt/svnindex.xsl 


  echo ============== add the Include of the $conf into $APACHE2_CONF if not there already
  grep $SVN_APACHE2_CONF $APACHE2_CONF  || $ASUDO echo "Include $SVN_APACHE2_CONF"  >> $APACHE2_CONF  


  echo ============= tail -10 $APACHE2_CONF
  tail -10 $APACHE2_CONF 

  [ "$APACHE2_HOME/sbin" == $(dirname $(which apachectl)) ] || (  echo your PATH to apache2 executables is not setup correctly  && return ) 
  

  apache2-setport $SVN_PORT

  apachectl configtest && echo restarting apache2 && apachectl restart || echo apachectl configtest failed


  #curl -o $APACHE2_HTDOCS/favicon.ico http://grid1.phys.ntu.edu.tw:6060/tracs/red/

}




svn-apache2-location(){


cat << EOC
#
#      do not edit, created by svn.bash::svn-apache2-location 
#       http://svnbook.red-bean.com/en/1.0/ch06s04.html
#
#   NB no need to restart apache2 on creating new repositories
#
#
<Location /repos>
      DAV svn
      SVNParentPath $SVN_PARENT_PATH
      SVNIndexXSLT /resources/xslt/svnindex.xsl

      # access policy file
      AuthzSVNAccessFile $SVN_APACHE2_AUTHZACCESS

      # try anonymous access first, resort to real 
	  # authentication if necessary.
      Satisfy Any
      Require valid-user

      # how to authenticate a user
      AuthType Basic
      AuthName "svn-repos"
      # users file
      AuthUserFile $SVN_APACHE2_AUTH

</Location>
EOC

}



svn-apache2-authzaccess(){

cat << EOA
#
#      do not edit, created by svn.bash::svn-apache2-authzaccess, svn-apache2-conf
#      http://svnbook.red-bean.com/en/1.0/ch06s04.html
#   
[groups]
dyw-admin = dayabaysoft
dyw-user = blyth, thho
dyw-novice = simon 

#  read access for everyone
[/]
* = r

[red:/trunk]
@dyw-novice = r
@dyw-user = rw
@dyw-admin = rw

# caution this matches the path in every repository 
[/trunk]
@dyw-novice = r
@dyw-user = rw
@dyw-admin = rw




EOA

}




svn-apache2-location-simpleauth-deprecated(){

## NB no need to restart apache2 on creating new repositories
cat << EOC
<Location /repos>
      DAV svn
      SVNParentPath $SVN_PARENT_PATH
      SVNIndexXSLT /resources/xslt/svnindex.xsl

      AuthType Basic
      AuthName "svn-repos"
      AuthUserFile $SVN_APACHE2_AUTH

# For any operations other than these, require an authenticated user.
      <LimitExcept GET PROPFIND OPTIONS REPORT>
	     Require valid-user
      </LimitExcept>

</Location>
EOC

}



svn-global-ignores(){

cat << EOI
#  uncomment global-ignores in [miscellany] section of
#     $HOME/.subversion/config
#  setting it to : 
#
global-ignores = setup.sh setup.csh cleanup.sh cleanup.csh Linux-i686* Darwin* InstallArea Makefile load.C
#
#    NB there is no whitespace before "global-ignores"
# 
#  after this   
#        svn status -u 
#  should give a short enough report to be useful
#
EOI

echo vi $HOME/.subversion/config


}








