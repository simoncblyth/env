export SCM_FOLD=/var/scm
#
#   after a change in SCM_FOLD in need to tell apache2 about the change with :
#      svn-apache2-conf
#      trac-apache2-conf
#      
#

SVN_NAME=subversion-1.4.0
SVN_ABBREV=svn
SVN_APACHE2_CONF=etc/apache2/svn.conf 
SVN_APACHE2_AUTH=etc/apache2/svn-apache2-auth
SVN_APACHE2_AUTHZACCESS=etc/apache2/svn-apache2-authzaccess
export SVN_PARENT_PATH=$SCM_FOLD/repos
export SVN_HOME=$LOCAL_BASE/$SVN_ABBREV/$SVN_NAME
export PATH=$SVN_HOME/bin:$PATH


PYTHON_NAME=Python-2.5.1
export PYTHON_HOME=$LOCAL_BASE/python/$PYTHON_NAME
export PATH=$PYTHON_HOME/bin:$PATH

SQLITE_NAME=sqlite-3.3.16
export SQLITE_HOME=$LOCAL_BASE/sqlite/$SQLITE_NAME
export LD_LIBRARY_PATH=$SQLITE_HOME/lib:$LD_LIBRARY_PATH

APACHE2_NAME=httpd-2.0.59
APACHE2_ABBREV=apache2
export APACHE2_HOME=$LOCAL_BASE/$APACHE2_ABBREV/$APACHE2_NAME
export PATH=$APACHE2_HOME/sbin:$PATH

ASUDO=$SUDO
export ASUDO=



# trac-admin is made available vis the PYTHON_HOME/bin PATH setting

scm-use-tracurl(){
	
   name=${1:-dummy}
    rev=${1:-0}	
    turl="http://$SCM_HOST:$SCM_PORT/tracs/$name/browser/trunk"

    [ "$rev" == "0" ] && echo $turl || echo "$turl?rev=$rev"

}


scm-use-create-local(){

   #   usage:
   #        scm-create-local name [path]
   #
   #     if the directory path is not specified, or is not a valid directory an empty repository 
   #        ... with just the trunk branches tags structure will be created
   #

   name=${1:-dummy}     ## name of the repository and tracitory to create
   path=${2:-dummy}     ## directory path to import, if a valid directory path
   
   
   
   [ "$SCM_FOLD/repos" == "$SVN_PARENT_PATH" ] || ( echo non-standard SCM layout ABORT && return )

   [ "$SVN_HOME/bin"    == $(dirname $(which svnadmin)) ]   || ( echo check your path , svnadmin from non-controlled location .... ABORT && return ) 
   [ "$SVN_HOME/bin"    == $(dirname $(which svn)) ]        || ( echo check your path , svn from non-controlled location .... ABORT && return ) 
   [ "$PYTHON_HOME/bin" == $(dirname $(which trac-admin)) ] || ( echo check your path , trac-admin from non-controlled location .... ABORT && return ) 

   repo=$SCM_FOLD/repos/$name
   tmpl=$PYTHON_HOME/share/trac/templates

   ## create the svn repository 

   mkdir -p $SCM_FOLD/repos                      ## this is SVN_PARENT_PATH
   svnadmin create  $SCM_FOLD/repos/$name
   
   ## do a first import into the repository 
   
   
   tmpdir=/tmp/scb/toimp$$
   mkdir -p $tmpdir/{branches,tags,trunk}

   if [ -d "$path" ]; then
      cp -r $path/ $tmpdir/trunk/
   fi
   
   svn import $tmpdir file://$repo -m "initial import from $path "
   rm -rf $tmpdir

   ## create the trac env to follow changes in the svn repository 
   
   mkdir -p $SCM_FOLD/tracs
   trac-admin $SCM_FOLD/tracs/$name initenv $name sqlite:db/trac.db svn $repo $tmpl

   ## prepare the trac env to be accessible thru apache2 with wsgi
   echo create apache folder within the trac site instance folder $SCM_FOLD/tracs/$name in which to put the .wsgi 
   $ASUDO mkdir -p  $SCM_FOLD/tracs/$name/apache

   ## Nb changes here must be made in tandem with modwsgi.bash::modwsgi-tracs-conf
   wsgi=$SCM_FOLD/tracs/$name/apache/$name.wsgi 
   echo writing wsgi $wsgi python app 
   $ASUDO modwsgi-use-app $name   >  $wsgi


   ## tweak the trac.ini for fine grained permissions
   trac-use-authz $name

}


scm-use-remove-local(){

   name=${1:-dummy}
   
   [ "$SCM_FOLD/repos" == "$SVN_PARENT_PATH" ] || ( echo non-standard SCM layout ABORT && return )

   [ -d "$SCM_FOLD/repos/$name" ] && echo removing $SCM_FOLD/repos/$name && rm -rf $SCM_FOLD/repos/$name
   [ -d "$SCM_FOLD/tracs/$name" ] && echo removing $SCM_FOLD/tracs/$name && rm -rf $SCM_FOLD/tracs/$name

}

