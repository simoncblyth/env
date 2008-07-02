



#
#   after a change in SCM_FOLD in need to tell apache2 about the change with :
#      svn-apache2-conf
#      trac-apache2-conf
#      
#


scm-use-env(){

   elocal-
   svn-
   #apache2-
   apache-
   
   python-
   sqlite-
   
   trac-use-

}






# trac-admin is made available vis the PYTHON_HOME/bin PATH setting

scm-use-tracurl(){
	
    local name=${1:-dummy}
    local rev=${1:-0}	
    local turl="$SCM_URL/tracs/$name/browser/trunk"

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
   #        scm-use-create-local test INIT
   #        scm-use-create-local test
   #
   #     if the directory path is not specified, or is not a valid directory an empty repository 
   #        ... with just the trunk branches tags structure will be created if INIT is supplied
   #
   #     note the ownership flipping ... for local access needs to belong to
   #     $USER .. for remote access thru apache2 needs to belong to the
   #     APACHE2_USER 
   #
   #     this flipping risks access to other repositories 
   #     as the update is being made 
   #     ... better to compartmentalize more 
   #

   local name=${1:-dummy}     ## name of the repository and tracitory to create
   local 
   path=${2:-EMPTY}      ## directory path to import, if a valid directory path
                        ## or INIT to just initialize with branches, trunk, tags
   
   echo ====== scm/scm-use.bash::scm-use-create-local name:$name path:$path starting  ====
   
    # on Leopard 
   if [ "$NODE_APPROACH" == "stock" ]; then
	  [ "/usr/bin"    == $(dirname $(which svnadmin)) ]   || ( echo check your path , svnadmin from non-controlled location .... ABORT && return ) 
      [ "/usr/bin"    == $(dirname $(which svn)) ]        || ( echo check your path , svn from non-controlled location .... ABORT && return ) 
	  [ "/usr/local/bin" == $(dirname $(which trac-admin)) ] || ( echo check your path , trac-admin from non-controlled location .... ABORT && return ) 
	  tmpl=
	  USER_GROUP=staff
   else
      [ "$SVN_HOME/bin"    == $(dirname $(which svnadmin)) ]   || ( echo check your path , svnadmin from non-controlled location .... ABORT && return ) 
      [ "$SVN_HOME/bin"    == $(dirname $(which svn)) ]        || ( echo check your path , svn from non-controlled location .... ABORT && return ) 
      [ "$PYTHON_HOME/bin" == $(dirname $(which trac-admin)) ] || ( echo check your path , trac-admin from non-controlled location .... ABORT && return ) 
	  tmpl=$PYTHON_HOME/share/trac/templates
      USER_GROUP=$USER
   fi
   
   
   if [ -d "$SCM_FOLD" ]; then
     echo =========  scm folder $SCM_FOLD exists already , temporarily adjusting ownership to USER $USER ... may need password
     sudo chown -R $USER:$USER_GROUP $SCM_FOLD
   else
     echo =========  creating scm folder $SCM_FOLD , owned by $USER , temporarily adjusting ownership to USER $USER ... may need password
	 sudo mkdir -p $SCM_FOLD
	 sudo chown -R $USER:$USER_GROUP $SCM_FOLD 
   fi
	   
   
   [ "$SCM_FOLD/repos" == "$SVN_PARENT_PATH" ] || ( echo non-standard SCM layout ABORT && return )


  
   repo=$SCM_FOLD/repos/$name
  
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
   
   if ( [ "X$path" == "XEMPTY" ]  ); then
       echo ====== path $path so making an empty repository, with revision zero 
   else
   
      tmpdir=/tmp/scb/toimp$$
      echo ======= create standard reposity layout beneath $tmpdir
      mkdir -p $tmpdir/{branches,tags,trunk} 
   
      if [ -d "$path" ]; then
         echo ======= copy sources to import into tmpdir $tmpdir 
         cp -r $path/ $tmpdir/trunk/
      else
         echo ======= the path supplied does not exist 
      fi
      
      echo ======= import from tmpdir $tmpdir into file://$repo 
      svn import $tmpdir file://$repo -m "initial import from $path "
      
      rm -rf $tmpdir
        
   fi
 
  
   echo ======== create trac env to follow changes in the svn repository 

   ## cannot do this directly another user such as ... www...  as gives Fatal Python error 

   #  Trac 0.11b1 ... the last tmpl argument is not there in this version (yep tis for clearsilver templates) 
   # initenv <projectname> <db> <repostype> <repospath>
   #	-- Create and initialize a new environment from arguments

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


  ## move the path setup for authzaccess from trac-use-authzaccess
  ##
  ##  on Tiger+Linux :
  ##    APACHE2_HOME            /usr/local/apache2/httpd-2.0.59    
  ##    SVN_APACHE2_AUTHZACCESS  etc/apache2/svn-apache2-authzaccess
  ## 
  ## on Leopard create a folder to hold local mods
  ##       /private/etc/apache2/local/  
  ## 
    
   authzaccess="$APACHE2_BASE/$SVN_APACHE2_AUTHZACCESS" 
 


   ## tweak the trac.ini for fine grained permissions
   echo ================== invoke trac-use-authz $name $authzaccess =====
   trac-use-authz $name $authzaccess
   echo ================== completed trac-use-authz $name =====
   
   echo  ================= setting chownership of SCM_FOLD $SCM_FOLD on NODE_TAG $NODE_TAG to APACHE2_USER $APACHE2_USER 
   echo  ================= this allows remote control of svn thru apache2 , may need the USER $USER password on node $NODE_TAG
   sudo chown -R $APACHE2_USER:$APACHE2_USER $SCM_FOLD


   echo ====== scm/scm-use.bash::scm-use-create-local completed  ====


}


scm-use-remove-local(){

   name=${1:-dummy}
   
   [ "$SCM_FOLD/repos" == "$SVN_PARENT_PATH" ] || ( echo non-standard SCM layout ABORT && return )

   [ -d "$SCM_FOLD/repos/$name" ] && echo removing $SCM_FOLD/repos/$name && $SUDO rm -rf $SCM_FOLD/repos/$name
   [ -d "$SCM_FOLD/tracs/$name" ] && echo removing $SCM_FOLD/tracs/$name && $SUDO rm -rf $SCM_FOLD/tracs/$name

}

