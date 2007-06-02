export SCM_FOLD=/var/scm
#
#   after a change in SCM_FOLD in need to tell apache2 about the change with :
#      svn-apache2-conf
#      trac-apache2-conf
#      
#

SVN_NAME=subversion-1.4.0
SVN_ABBREV=svn

## just needed by svn.bash
SVN_APACHE2_CONF=etc/apache2/svn.conf

## these are needed by both SVN + Trac  
SVN_APACHE2_AUTH=etc/apache2/svn-apache2-auth
SVN_APACHE2_AUTHZACCESS=etc/apache2/svn-apache2-authzaccess
export SVN_PARENT_PATH=$SCM_FOLD/repos
export SVN_HOME=$LOCAL_BASE/$SVN_ABBREV/$SVN_NAME

export DYLD_LIBRARY_PATH=$SVN_HOME/lib/svn-python/svn:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$SVN_HOME/lib/svn-python/libsvn:$DYLD_LIBRARY_PATH

export PATH=$SVN_HOME/bin:$PATH


PYTHON_NAME=Python-2.5.1
export PYTHON_HOME=$LOCAL_BASE/python/$PYTHON_NAME
export PYTHON_SITE=$PYTHON_HOME/lib/python2.5/site-packages
export PATH=$PYTHON_HOME/bin:$PATH

SQLITE_NAME=sqlite-3.3.16
export SQLITE_HOME=$LOCAL_BASE/sqlite/$SQLITE_NAME
export LD_LIBRARY_PATH=$SQLITE_HOME/lib:$LD_LIBRARY_PATH

APACHE2_NAME=httpd-2.0.59
APACHE2_ABBREV=apache2

if [ "$NODE_TAG" == "G" ]; then
   APACHE2_USER=www
elif [ "$NODE_TAG" == "H" ]; then
   APACHE2_USER=apache
else
   APACHE2_USER=apache
fi

export APACHE2_HOME=$LOCAL_BASE/$APACHE2_ABBREV/$APACHE2_NAME
APACHE2_ENV=$APACHE2_HOME/sbin/envvars
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

scm-use-test(){

  cd 
  TSUDO="sudo -u www"
   scm-use-create-local test Desktop/kambiu-ten-pages
}


scm-use-create-local(){

   #   usage:
   #        scm-use-create-local name [path]
   #
   #        scm-use-create-local test Desktop/kambiu-ten-pages
   #
   #     if the directory path is not specified, or is not a valid directory an empty repository 
   #        ... with just the trunk branches tags structure will be created
   #
   #     note the ownership flipping ... for local access needs to belong to
   #     $USER .. for remote access thru apache2 needs to belong to the
   #     APACHE2_USER 
   #
   #     this flipping risks access to other repositories 
   #     as the update is being made 
   #     ... better to compartmentalize more 
   #

   name=${1:-dummy}     ## name of the repository and tracitory to create
   path=${2:-dummy}     ## directory path to import, if a valid directory path
   
   if [ -d "$SCM_FOLD" ]; then
     echo =========  scm folder $SCM_FOLD exists already , temporarily adjusting ownership to USER $USER ... may need password
     sudo chown -R $USER:$USER $SCM_FOLD
   else
     echo =========  creating scm folder $SCM_FOLD , owned by $USER , temporarily adjusting ownership to USER $USER ... may need password
	 sudo mkdir -p $SCM_FOLD
	 sudo chown -R $USER:$USER $SCM_FOLD 
   fi
	   
   
   [ "$SCM_FOLD/repos" == "$SVN_PARENT_PATH" ] || ( echo non-standard SCM layout ABORT && return )

   [ "$SVN_HOME/bin"    == $(dirname $(which svnadmin)) ]   || ( echo check your path , svnadmin from non-controlled location .... ABORT && return ) 
   [ "$SVN_HOME/bin"    == $(dirname $(which svn)) ]        || ( echo check your path , svn from non-controlled location .... ABORT && return ) 
   [ "$PYTHON_HOME/bin" == $(dirname $(which trac-admin)) ] || ( echo check your path , trac-admin from non-controlled location .... ABORT && return ) 

   repo=$SCM_FOLD/repos/$name
   tmpl=$PYTHON_HOME/share/trac/templates

   ## create the svn repository , owned by root if SUDO is set
 
   if [ -d "$repo" ]; then
     echo =========  repository folder $repo exists already
   else	   
     echo ========= creating svn repository at $repo  
     mkdir -p $SCM_FOLD/repos   || ( echo scm-use-create-local ABORT && return  )
     svnadmin create  $repo     || ( echo scm-use-create-local ABORT && return  )
   fi
  
   # svnadmin doesnt exit with an error when it should ... eg for permission denied issues
   # do a first import into the repository 
   
   echo ======= copy sources to import into tmpdir $tmpdir
   
   tmpdir=/tmp/scb/toimp$$
   mkdir -p $tmpdir/{branches,tags,trunk}

   if [ -d "$path" ]; then
      cp -r $path/ $tmpdir/trunk/
   fi
 
   
   echo ======= import from tmpdir $tmpdir into file://$repo 
   svn import $tmpdir file://$repo -m "initial import from $path "
   rm -rf $tmpdir

  
   echo ======== create trac env to follow changes in the svn repository 

   ## cannot do this directly another user such as ... www...  as gives Fatal Python error 

   mkdir -p $SCM_FOLD/tracs
   echo trac-admin $SCM_FOLD/tracs/$name initenv $name sqlite:db/trac.db svn $repo $tmpl
        trac-admin $SCM_FOLD/tracs/$name initenv $name sqlite:db/trac.db svn $repo $tmpl

   ## prepare the trac env to be accessible thru apache2 with wsgi
   if [ "$SCM_FRONTEND" == "wsgi" ]; then
      echo =========== create apache folder within the trac site instance folder $SCM_FOLD/tracs/$name in which to put the .wsgi 
      mkdir -p  $SCM_FOLD/tracs/$name/apache

      ## Nb changes here must be made in tandem with modwsgi.bash::modwsgi-tracs-conf
      #   are using mod_python so not needed
 
      wsgi=$SCM_FOLD/tracs/$name/apache/$name.wsgi 
      echo writing wsgi $wsgi python app 
       modwsgi-use-app $name   >  $wsgi
   fi

   ## tweak the trac.ini for fine grained permissions
   trac-use-authz $name
   
   echo  ================= setting chownership of SCM_FOLD $SCM_FOLD on NODE_TAG $NODE_TAG to APACHE2_USER $APACHE2_USER 
   echo  ================= this allows remote control of svn thru apache2 , may need the USER $USER password on node $NODE_TAG
   sudo chown -R $APACHE2_USER:$APACHE2_USER $SCM_FOLD

}


scm-use-remove-local(){

   name=${1:-dummy}
   
   [ "$SCM_FOLD/repos" == "$SVN_PARENT_PATH" ] || ( echo non-standard SCM layout ABORT && return )

   [ -d "$SCM_FOLD/repos/$name" ] && echo removing $SCM_FOLD/repos/$name && $SUDO rm -rf $SCM_FOLD/repos/$name
   [ -d "$SCM_FOLD/tracs/$name" ] && echo removing $SCM_FOLD/tracs/$name && $SUDO rm -rf $SCM_FOLD/tracs/$name

}

