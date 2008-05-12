#
#
#   this is needs generalization to handle all webapps :
#       - exist(jetty)
#       - chiba(tomcat)
#       - ..? 
#   as they all share the namespace and portspace and apache2 config space
#
##############################################################################
#
#
#   exist-get 
#   exist-install     installs in duplicate
#
#   exist-populate   workflow_0
#   exist-populate   workflow_1
#   exist-populate   exist_0
#
#   sudo apachectl stop
#
#   exist-ctl workflow_0 start
#   exist-ctl workflow_1 start
#   exist-ctl exist_0    start
#
#   ........ note bizarreness wrt process start ordering , probably best to turn "apachectl  stop"
#  while starting workers 
#
#   this is possible due to my current non-dynamic modjk config ... which has non-operational workers
#
#
#
#
#    19 May 2007 23:18:04,272 [main] WARN  (WebApplicationContext.java [resolveWebApp]:259) - Web application not found ../../webapp
#   ... when forget to do: exist-jetty-startup-customize
#
#
#  have not succeded to isolate instances ... clash of logs :
#
# Only in 1/tools/jetty/logs: 2007_05_19.request.log
# Only in 1/tools/jetty/work: Jetty__7070__workflow_0
# Only in 1/tools/jetty/work: Jetty__9090__workflow_1
# Only in 1/webapp/WEB-INF/logs: cocoon-ehcache.log
# Only in 1/webapp/WEB-INF/logs: exist.log
# Only in 1/webapp/WEB-INF/logs: profile.log
# Only in 1/webapp/WEB-INF/logs: validation.log
# Only in 1/webapp/WEB-INF/logs: xacml.log
# Only in 1/webapp/WEB-INF/logs: xmldb.log
#
#
#  migration....    with an eye to minimising the pain next time 
#
# [g4pb:/usr/local/exist/live/workflow/1/webapp] blyth$ grep \<document\  *.xml
# examples.xml:<document xmlns:xi="http://www.w3.org/2001/XInclude">
# facts.xml:<document xmlns:xi="http://www.w3.org/2001/XInclude">
# index.xml:<document xmlns:xi="http://www.w3.org/2001/XInclude">
#
#
#
#   destined to replace ...
#       /usr/local/heprez/src/exist/build.xml
#
#
#    is would be nice  to have a single exist code base which can be
#    used for multiple instances ... distinguishing by means of different
#    configuration files that are pointed to by system properties on
#    invokation ... unfortunately although jetty supports the conf file as 
#    and argument , exists start.jar org/exist/start/Main.java does not pass
#    arguments thru to jetty , but rather constructs the absolute path to 
#    jetty.xml from exist.home ... which can be passed in -Dexist.home=blah
#
#    can circumvent this to some extent by editing the jetty conf file prior to launch ... using 
#    the 0 to 1 swap technique
#
#    BUT it seems that the database position is not moved by -Dwebapp.home
#
#
# File lock last access timestamp: May 19, 2007 /usr/local/exist/eXist-1.1.1-newcore-build4311/1/webapp/WEB-INF/data/dbx_dir.lck
# configuration error: The database directory seems to be locked by another database instance. Found a valid lock file: /usr/local/exist/eXist-1.1.1-newcore-build4311/1/webapp/WEB-INF/data/dbx_dir.lck
# org.exist.EXistException: The database directory seems to be locked by another database instance. Found a valid lock file: /usr/local/exist/eXist-1.1.1-newcore-build4311/1/webapp/WEB-INF/data/dbx_dir.lck
# at org.exist.storage.BrokerPool.canReadDataDir(BrokerPool.java:611)
# at org.exist.storage.BrokerPool.<init>(BrokerPool.java:563)
# at org.exist.storage.BrokerPool.configure(BrokerPool.java:169)
# at org.exist.storage.BrokerPool.configure(BrokerPool.java:143)
# at org.exist.JettyStart.run(JettyStart.java:90)
# at org.exist.JettyStart.main(JettyStart.java:48)
# at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
# at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:39)
# at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:25)
# at java.lang.reflect.Method.invoke(Method.java:585)
# at org.exist.start.Main.invokeMain(Main.java:128)
# at org.exist.start.Main.run(Main.java:405)
# at org.exist.start.Main.main(Main.java:59)
#
#
#   observation shows that using -Dwebapp.home=  means the webapp contents is coming from elsewhere 
#   but the database is not relocated ...
#
#
#   maybe handle by on-the-fly setting of db-connection/@files changing from
#
#
#
#
#


exist-env(){

EXIST_NIK=exist
EXIST_REP=https://exist.svn.sourceforge.net/svnroot/exist

EXIST_DIST=http://prdownloads.sourceforge.net/exist
#EXIST_DIST=http://jaist.dl.sourceforge.net/sourceforge/exist
EXIST_DIST=http://nchc.dl.sourceforge.net/sourceforge/exist
EXIST_NAME=eXist-1.1.1-newcore-build4311


export EXIST_FOLD=$LOCAL_BASE/$EXIST_NIK

}


exist-ctx(){

   tag=$EXIST_NAME 
   nik=$EXIST_NIK
   rep=$EXIST_REP

   [ -d $LOCAL_BASE/$nik  ] || ( cd $LOCAL_BASE && sudo mkdir -p $nik && sudo chown -R $USER $nik )
   cd $LOCAL_BASE/$nik

   if [ "$tag" == "head" ]; then
      
	  branch=trunk/eXist
	  if [ -L "${nik}_head" ]; then
		 tag=$(readlink ${nik}_head) 
	  else	  
         tag=${nik}_$(base-datestamp "now")
	  fi	 
      [ -d $tag  ] &&  cd $tag || echo tag folder $tag not checked out yet 

   else
      jar=$tag.jar
	  url=$EXIST_DIST/$jar
	  [ -d $tag/0 ] || mkdir -p $tag/0 
	  [ -d $tag/1 ] || mkdir -p $tag/1 
   fi
   
}


exist-get(){
   exist-ctx $*
   [ -f $jar ] || curl -o $jar $url
}

exist-wipe(){
  exist-ctx $*
  rm -rf $tag/0 
  rm -rf $tag/1
}


exist-install(){
   exist-ctx $*

   ## need to pass absolute path ... for the startup scripts to auto-set "EXIST_HOME" approprioately 

   java -jar $jar -p $EXIST_FOLD/$tag/0
   java -jar $jar -p $EXIST_FOLD/$tag/1

   exist-jetty-startup-customize
}


exist-jetty-startup-customize(){

   exist-ctx $*

  # moving the $* from after the jetty to before the -jar 
  #	this allows extra -D switches to be passed in from the invokation cmdline 
  # also switch the EXIST_HOME as are using the zero version as the source for the tranformation into the home directory 
  # 
  #
  #  needs to be done once only 
 
  local in=${tag}/0/bin/startup.sh
  local ou=${tag}/1/bin/startup.sh
	 
  perl -p -e 's/\$JAVA_ENDORSED_DIRS/\$JAVA_ENDORSED_DIRS \$\*/ || s/jetty \$\*/jetty/ || s@${tag}/0@${tag}/1@' $in > $ou
  chmod u+x $ou
  diff -w $in $ou 

}




exist-migrate-once-only(){

  exist-ctx

  local wkr=${1:-workflow_1}
  local ctx=$(exist-lookup context $wkr)
  local idx=$(exist-lookup index $wkr)
  
  local oldwebapp=$HOME/wf 
  local newwebapp=live/$ctx/$idx
  cd $newwebapp


  ## this is echo protected ... to prevent accidental use

  echo=echo 

  ## remove the default content , intro quickstart etc...
  $echo rm -f webapp/*.xml

  ## copy in the top level items without counterparts into the new webapp
  for item in $oldwebapp/* 
  do
	 if [ -d "$item" ]; then 
	     fold=$(basename $item)
	     dest=webapp/$fold
	     if [ -d "$dest" ]; then
		     echo skip preexisting folder $dest
	     else
		     cmd="cp -r $item $dest"
			 echo $cmd
			 $echo $cmd
	     fi
     elif [ -f "$item" ]; then 
 
        name=$(basename $item)
        dest=webapp/$name
	    if [ -f "$dest" ]; then
			echo untested branch ...  
			cmd="mv $dest $dest.original && cp $name $dest"
			echo $cmd
			$echo $cmd
	    else
			cmd="cp $name $dest"
			echo $cmd
			$echo $cmd
        fi			
     else
		 echo huhhh $item
	 fi
  done	  

#
# skip preexisting file webapp/sitemap.xmap
#
# skip preexisting folder webapp/WEB-INF
# skip preexisting folder webapp/resources
# skip preexisting folder webapp/stylesheets
# skip preexisting folder webapp/xquery
#
#
#


}





exist-customize(){
   exist-jetty-conf-customize $*
   exist-jetty-startup-customize $*
}

exist-instance(){
   exist-ctx 
   ctx=${1:-workflow}
   mkdir -p live/$ctx/{0,1}
}

exist-populate(){
   exist-ctx $*
   wkr=${1:-dummycontext_dummyindex}

   ctx=$(exist-lookup context $wkr)
   idx=$(exist-lookup index $wkr)
  port=$(exist-lookup port $wkr)
 
   if [ -d live/$ctx/$idx/webapp ]; then
	   echo cannot exist-populate a preexisting folder ... for source protection 
   else 

	   echo mkdir -p live/$ctx/$idx
	        mkdir -p live/$ctx/$idx
       echo cp -r $EXIST_FOLD/$EXIST_NAME/0/webapp live/$ctx/$idx
            cp -r $EXIST_FOLD/$EXIST_NAME/0/webapp live/$ctx/$idx
   fi	   
}


exist-ports(){
   wkr=${1:-workflow_0}
   
  #   exist_0) ports="8080:8009" ;;
  #  workflow_1) ports="9090:9009" ;;
  
    case $wkr in 
         chiba_0) ports="8080:8009" ;;
         hfagc_0) ports="9090:9009" ;;
        legacy_0) ports="7070:7009" ;;
      workflow_0) ports="7070:7009" ;;
      workflow_1) ports="9090:9009" ;;
    		   *) ports="?:?"       ;;
   esac	 
   echo $ports
}

exist-lookup(){

   ## hmm would be easier to do this with an sqlite3 table

   local typ=${1:-port}
   local wkr=${2:-workflow_0}
   local ctx=$( echo $wkr | cut -d _ -f 1)
   local idx=$( echo $wkr | cut -d _ -f 2)
   local uctx=$wkr 
   
   if [ "$ctx" == "legacy" ]; then
       uctx="workflow"
   elif [ "$ctx" == "chiba" ]; then
       uctx="chiba"
   elif [ "$ctx" == "hfagc" ]; then
       uctx="hfagc"    
   fi	 
 
   if [ "$typ" == "workers" ]; then
      #echo "workflow_0 workflow_1 exist_0"
      # this has to match the above manually, ughhh
      echo "workflow_0 workflow_1 chiba_0 hfagc_0"
      
   elif [ "$typ" == "context" ]; then
      
      if [ "$ctx" == "legacy" ]; then
		 echo "workflow"
	  else	 
         echo $ctx 
	  fi
      	 
   elif [ "$typ" == "ucontext" ]; then
      echo $uctx 
   elif [ "$typ" == "index" ]; then
      echo $idx 
   elif [ "$typ" == "home" ]; then
   
      if [ "$ctx" == "legacy" ]; then
		  echo  "/usr/local/jars/xist/live/eXist-snapshot-20060316/2/eXist-snapshot-20060316" 
	  elif [ "$ctx" == "hfagc" ]; then
          echo "/usr/local/heprez/install/exist/eXist-snapshot-20051026/unpack/2"
      else
          echo $EXIST_FOLD/$EXIST_NAME/1
      fi 		  

   elif [ "$typ" == "livedir" ]; then
   
      if [ "$ctx" == "legacy" ]; then
		  echo "/usr/local/jars/workflow/live/2"
	  elif [ "$ctx" == "hfagc" ]; then
          echo "/tmp" 
      else	  
	     echo $LOCAL_BASE/exist/live/$ctx/$idx
      fi		 
   
   elif ([ "$typ" == "port" ] || [ "$typ" == "jkport" ] || [ "$typ" == "uri" ] ); then
	 
      local ports=$(exist-ports $wkr)
      local port=$(echo $ports | cut -d : -f 1)
      local jkport=$(echo $ports | cut -d : -f 2)

      case $typ in
	     port) echo $port ;;
	   jkport) echo $jkport ;;
          uri) echo "xmldb:exist://localhost:$port/$uctx/xmlrpc"
	  esac
   fi
}

exist-ctl(){

   local wkr=${1:-dummyworker_0}	
   local cmd=${2:-dummy}

   echo exist-ctl wkr:$wkr cmd:$cmd

   local ctx=$(exist-lookup context $wkr)
   local idx=$(exist-lookup index $wkr)
   local port=$(exist-lookup port $wkr)
   local livedir=$(exist-lookup livedir $wkr)
   local existhome=$(exist-lookup home $wkr)
   local uctx=$(exist-lookup ucontext $wkr)
   local uri=$(exist-lookup uri $wkr)

   echo ctx:$ctx idx:$idx port:$port uctx:$uctx

   if [ "$cmd" == "startup" ]; then
      
	  ## setup the ports, context  etc.. in jetty.xml
      exist-jetty-customize "${ctx}_${idx}"
      
	  ## setup the folder to keep the database files and journal files
      exist-conf-customize "${ctx}_${idx}"
	  
      ## ensure the script is not confused by EXIST_HOME in the environment 
      unset EXIST_HOME 
   fi


   [ -L "$EXIST_FOLD/backup/restore-wkr" ] || ( echo  the source worker to be restored into the current worker must be specified by a link ... $EXIST_FOLD/backup/restore-wkr  && return 1 ) 
   local restorewkr=$(readlink $EXIST_FOLD/backup/restore-wkr)

   local dcmd="echo noop"
   case $cmd in 
	        start) dcmd="$existhome/bin/startup.sh -Dwebapp.home=$livedir" ;;
             stop) dcmd="$existhome/bin/shutdown.sh  -l xmldb:exist://localhost:$port/$uctx/xmlrpc" ;;
		   backup) dcmd="exist-full backup  $existhome/bin/backup.sh $EXIST_FOLD/backup/$wkr $uri" ;;
		  restore) dcmd="exist-full restore $existhome/bin/backup.sh $EXIST_FOLD/backup/$restorewkr  $uri" ;; 
                *) dcmd="echo exist-ctl only handles start and stop" ;;
   esac

   echo $dcmd
   $dcmd
  
  
}



exist-full(){

    local operation=$1
    local script=$2
    local folder=$3
    local uri=$4

    #
    # smth fummy with --password "" ... when present the command doesnt work and get no error , maybe when havent set a password yet ??
    # having it pressent causes : 
    #     "ERROR: Unable to parse first argument for option -o "
    #
    #
    iwd=$(pwd)

	[ -d $folder ] || ( echo exist-full error folder $folder doesnt exist && return 1 ) 
    cd $folder 

	if [ "$operation" == "backup" ]; then
		
       local stamp=$(base-datestamp 'now')  
	   local cmd=" $script --user admin --backup /db --dir $stamp  --option uri=$uri " 
	   mkdir -p $stamp &&  echo $cmd && $cmd && rm -f last-backup && ln -s $stamp last-backup
	   
	elif [ "$operation" == "restore" ]; then
	
       ## not so trivial when migrating ... as want to cross instances ???
	   local stamp=$(readlink last-backup) 
	   cmd=" $script --user admin --restore  $folder/$stamp/db/__contents__.xml  --option uri=$uri " 
	   echo $cmd && $cmd && rm -f last-restore && ln -s $stamp last-restore

	else
		
		echo operation $operation not supported && return 1 

	fi
	pwd
	ls -alst .
	cd $iwd
}


exist-jkmount(){

  ## hmm this is exist specific ... could be tomcat 
  ## ... can handle in the lookup and branch here 

  local wkr=${1:-dummyworker}

  local ctx=$(exist-lookup context $wkr)
  local uctx=$(exist-lookup ucontext $wkr)
  local idx=$(exist-lookup index $wkr)

  local webapp=$LOCAL_BASE/exist/live/$ctx/$idx/webapp 

cat << EOW
# sourced from modjk-context
<IfModule mod_jk.c>
JkMount  /$uctx/*    $wkr 
</IfModule>

#Alias /$uctx/logo.jpg           $webapp/resources/logo.jpg
#Alias /$uctx/styles/            $webapp/resources/styles/
#Alias /$uctx/stylesheets/       $webapp/resources/stylesheets/
#Alias /stylesheets/system/      $webapp/resources/stylesheets/system/

EOW

}


exist-conf-customize(){

   exist-ctx $*
   local wkr=${1:-dummycontext_dummyindex}

   local ctx=$(exist-lookup context $wkr)
   local uctx=$(exist-lookup ucontext $wkr)
   local idx=$(exist-lookup index $wkr)
   local port=$(exist-lookup port $wkr)
   local jkport=$(exist-lookup jkport $wkr)
   local livedir=$(exist-lookup livedir $wkr)

   echo exist-conf-customize wkr:$wkr ctx:$ctx idx:$idx port:$port jkport:$jkport
   
   local in=${tag}/0/conf.xml
   local ou=${tag}/1/conf.xml

   local cmd="xsltproc \
      --stringparam livedir "$livedir" \
	  --output $ou \
	  $XML_HOME/conf.xsl \
	  $in"

   echo $cmd
   $cmd

   diff -w $in $ou

}


exist-jetty-customize(){

   ## this needs to be done before every startup ... 
   ##  as will rewite the file for different workers

   exist-ctx $*
   local wkr=${1:-dummycontext_dummyindex}

   local ctx=$(exist-lookup context $wkr)
   local uctx=$(exist-lookup ucontext $wkr)
   local idx=$(exist-lookup index $wkr)
   local port=$(exist-lookup port $wkr)
   local jkport=$(exist-lookup jkport $wkr)

   echo exist-jetty-customize-worker wkr:$wkr ctx:$ctx idx:$idx port:$port jkport:$jkport

   ## NB cannot just grab a pristine copy from the jar , due to the use of izpack 
   
   local in=${tag}/0/tools/jetty/etc/jetty.xml
   local ou=${tag}/1/tools/jetty/etc/jetty.xml

   local cmd="xsltproc \
      --stringparam port $port \
	  --stringparam jkport $jkport \
	  --stringparam context $uctx \
	  --output $ou \
	  $XML_HOME/jetty.xsl \
	  $in"

	echo $cmd  
    $cmd
   
    diff -w $in $ou 

}





-exist-svn-get(){

   ## http://sourceforge.net/svn/?group_id=17691 
   ## http://exist.svn.sourceforge.net/viewvc/exist/trunk/eXist/

   exist-cd $*
  
   /usr/local/bin/svn co $rep/$branch $tag  && ln -s $tag ${nik}_head 

   #
   #   NB have to use /usr/local/bin/svn ... as :
   #
   #    /usr/local/svn/subversion-1.4.0/bin/svn
   #     was not built with SSL support ...
   #
   #     when trying to checkout/update  https://   get:
   #
   #         "svn: SSL is not supported"
   #
   #

}







