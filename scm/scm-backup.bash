


scm-backup-usage(){
cat << EOU

   \$SCM_FOLD   : $SCM_FOLD
   \$BACKUP_TAG : $BACKUP_TAG

   scm-backup-du   :   local  backup .gz sizes  in \$SCM_FOLD 
   scm-backup-rls  :   remote ls the .gz on the paired backup node $BACKUP_TAG
   scm-backup-mail :   send mail with the remote list  
    
   scm-backup-postfix-start  
    
    
   scm-backup-all-as-root :   does the below as root ... as done in the crontab
    
   scm-backup-all :   invokes the below 
      scm-backup-repo
      scm-backup-trac
      scm-backup-folder   for the apache-confdir   
      scm-backup-purge   : retain the backups from the last 7 days only
      
   scm-recover-all  <fromnode>
   
      NB the folders are not recovered by this, as is installation
      specific, nevertheless tis important that the users file is
      backed up
  
   scm-recover-folders <fromnode>
       still experimental .. NEEDS FURTHER CHECKING PRIOR TO REAL USAGE
       
       recovers the users and permissions files from the last backup
  
  
  
   scm-backup-rsync :   to the paired node   


   scm-backup-rsync-from-node : 
                   rsync the backups from a remote node 

   scm-backup-dybsvn-from-node : 
                  copy over the reps for a specific day 


  Common issues ...
  
     1) backups stopped :
    
    compare :
        scm-backup-du
        scm-backup-rls
    check base/cron.bash ... usually some environment change has broken the env setup for cron
    after modifications reset the cron backups..
    
       cron-
       cron-usage
       cron-backup-reset
       cron-list root
       cron-list blyth
   
    2) backups done but not synced off box
     
     
     Probably the agent needs restarting.. this is needs to be done manually after a reboot
     see 
        ssh--usage
        ssh--agent-start
     then check offbox passwordless access with
        scm-backup-
        scm-backup-rls
      
         
  Do an emergency backup and rsync, with :
  
    scm-backup-all-as-root 
    scm-backup-rsync       
    scm-backup-rls      ## check the remote tgz


  TODO : 
  
     1) divided reposnsibilities between here and cron.bash is a mess
     2) not easy to add things to crontab because of this morass 



EOU

}

scm-backup-env(){
   elocal-
   python-
   apache-
}


scm-backup-du(){
    find $SCM_FOLD -name '*.gz' -exec du -h {} \;
}


scm-backup-all-as-root(){

  sudo bash -lc "export HOME=$HOME ; export ENV_HOME=$HOME/env ; . $ENV_HOME/env.bash ; env- ; scm-backup- ; scm-backup-all  "

}

scm-backup-postfix-start(){

  sudo postfix start 
}



scm-backup-all(){
   
   local stamp=$(base-datestamp now %Y/%m/%d/%H%M%S)
   local base=$SCM_FOLD/backup/$LOCAL_NODE
   
   for path in $SCM_FOLD/repos/*
   do  
       if [ -d $path ]; then 
          local name=$(basename $path)
          scm-backup-repo $name $path $base $stamp        
       else
  	      echo === scm-backup-all repo === skip non-folder $path 
  	   fi
   done
   
   for path in $SCM_FOLD/tracs/*
   do  
       if [ -d $path ]; then 
           local name=$(basename $path)
           scm-backup-trac $name $path $base $stamp  
  		else
  		   echo === scm-backup-all trac === skip non-folder $path
  		fi
   done
   
   svn-
   
   local dir=$(svn-setupdir)
   local name=$(basename $dir)
   scm-backup-folder $name $dir $base $stamp
   
   scm-backup-purge $LOCAL_NODE
}



scm-recover-all(){

   # 
   # recovers both svn repositories and corresponding tracitories, from backup tarballs 
   #
   
   local fromnode=${1:-dummy}
   [ "$fromnode" == "dummy" ] && echo scm-recover-all needs a fromnode argument && return 1 
   
   
   
   local types="repos tracs"
   for type in $types
   do
      
      local base=$SCM_FOLD/backup/$fromnode/$type
      local dest=$SCM_FOLD/$type
      local user=$(apache-user)
      
      [ ! -d $dest ] && $SUDO mkdir -p $dest && [ "$SUDO" != "" ] && $SUDO chown $user:$user $dest       
      
      for path in $base/*
      do  
	      if [ -d $path ]; then 
             local name=$(basename $path)
             scm-recover-repo $name $path $dest
		  else
		     echo === scm-recover-all skip non-folder $path 
		  fi   
		  #
		  #  eg:
		  #    name : "workflow" 
		  #    path : /var/scm/backup/g4pb/tracs/workflow   or repos equivalent    
          #    dest : /var/scm/tracs or /var/scm/repos 
	  done
      
   done 

}


scm-recover-folders(){
  
   local msg="=== $FUNCNAME :"
   local fromnode=${1:-dummy}
   [ "$fromnode" == "dummy" ] && echo scm-recover-all needs a fromnode argument && return 1
   
   local base=$SCM_FOLD/backup/$fromnode/folders
   for path in $base/*
   do
      if [ -d $path ]; then
         local name=$(basename $path)
         local dest=$(scm-recover-destination $name)
         [ -z $dest ]   && echo $msg ABORT no destination for name $name path $path && return 1
         
             mkdir -p $dest   ## TESTING ONLY
             
         [ ! -d $dest ] && echo $msg ABORT dest $dest does not exist    && return 1  
     
         scm-recover-repo $name $path $dest 
      else
         echo $msg  skip non-folder $path  
      fi
  done

}

scm-recover-destination(){
  case $1 in 
         local|svnsetup|apache2) echo /tmp/$FUNCNAME/$(dirname $(svn-setupdir)) ;;
  esac  
  ## local name still in use on G, apache2 on H,  svnsetup elsewhere 
}





scm-backup-purge(){

  #
  #   deletes backup tgz and containing stamp folders   
  #   such that nmax remain for each repository and tracitory
  #
  #   bash array handling reference : http://tldp.org/LDP/abs/html/arrays.html
  #

  local node=${1:-$LOCAL_NODE} 
  local nmax=7
  local name
  local tgzs
  local itgz
  local ntgz 
  
  ## the bash version on hfag dies, if this is inside the for loop
  declare -a tgzs


  echo ======= scm-backup-purge =====   

  for path in $SCM_FOLD/backup/$node/{tracs,repos,folders}/* 
  do
     cd $path 
     
     name=$(basename $path)
     tgzs=($(find . -name '*.tar.gz'))
     ntgz=${#tgzs[@]}
     
     echo path:$path name:$name ntgz:$ntgz nmax:$nmax
     itgz=0
     while [ "$itgz" -lt "$ntgz" ]
     do    
        local tgz=${tgzs[$itgz]}
        if [ $(( $ntgz - $itgz > $nmax )) == 1 ]; then 
           local container=$(dirname $tgz) 
           local cmd="rm -rf $container"
           echo delete $tgz ... $cmd
           eval $cmd 
        else
           echo retain $tgz
        fi 
        let "itgz = $itgz + 1"
     done
     
  done
}

scm-backup-rls(){

   local target_tag=${1:-$BACKUP_TAG}
   
   if [ "X$target_tag" == "X" ]; then
      echo no paired backup node has been defined for node $LOCAL_NODE
   else
   
     local target_var=$(scm-backup-target $target_tag)
     local remote=$target_var/scm/backup 
	 ssh $target_tag "find $remote -name '*.gz' -exec du -hs {} \;"
  fi
}


scm-backup-mail(){

  local msg="=== $FUNCNAME :"
  local rls=/tmp/$FUNCNAME.txt
  
  echo $msg writing to $rls
  scm-backup-rls > $rls
  
  echo $msg sendmail $rls
  python-
  python-sendmail $rls

}




scm-backup-target(){
   local target_tag=${1:-$BACKUP_TAG}
   if [ "$target_tag" != "$BACKUP_TAG" ]; then 
      local vname=VAR_BASE_$target_tag 
      eval _VAR_BASE_BACKUP=\$$vname
      target_var=${_VAR_BASE_BACKUP:-$VAR_BASE_U}
      #echo ======== scm-backup-rsync to non-default target_tag:$target_tag  target_var:$target_var
   else
      target_var=$VAR_BASE_BACKUP
   fi
   echo $target_var
}


scm-backup-dir(){
   local tag=${1:-H}
   local node=$(local-tag2node $tag)
   local dir=$(local-scm-fold $tag)/backup  
   echo $dir
}


scm-backup-rsync-from-node(){

   local msg="# === $FUNCNAME : "
   local tag=$1
   [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT tag $tag is the same as current NODE_TAG $NODE_TAG ... ABORT && return 1
   
   local tgt=$(scm-backup-dir $NODE_TAG)
   mkdir -p $tgt
   local cmd="rsync -e ssh --delete-after -razvt $tag:$(scm-backup-dir $tag)/ $tgt/ "
   echo $cmd

}

scm-backup-dybsvn-from-node(){

   local msg="# === $FUNCNAME : "
   local tag=${1:-C}
   local dstamp="2008/07/31/122149"
   local stamp=${2:-$dstamp}
   local name="dybsvn"
   local orig="hfag"
   
   [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT tag $tag is the same as current NODE_TAG $NODE_TAG ... ABORT && return 1
     
   local loc=$(scm-backup-dir $NODE_TAG)  
   local rem=$(scm-backup-dir $tag)
   local reps=$(ssh $tag "ls -1 $rem/$orig/{repos,tracs}/$name/$stamp/$name*.tar.gz ")
   
   echo reps $reps
   
   for rep in $reps
   do
     
      local rel=${rep/$rem\//}
      local tgz=$loc/$rel
      
      echo rep $rep rel $rel tgz $tgz 
      mkdir -p $(dirname $tgz)
      
      local cmd="scp $tag:$rep $tgz"
      echo $cmd
      eval $cmd
   done

   cd $loc/$orig
   for dir in "tracs/$name repos/$name"
   do 
      ln -sf $stamp last 
   done
   

   
}





scm-backup-rsync(){

   # 
   # rsync the local backup repository to an off box mirror on the paired $BACKUP_TAG node 
   #   - have to set up ssh keys to allow non-interactive sshing 
   # 
   #  hmm the passwordless ssh is not setup for "root" user , so have to do this as me, but the above backup as root
   #

   local msg="=== $FUNCNAME :" 
   local target_tag=${1:-$BACKUP_TAG}   
  
   if [ "X$target_tag" == "X" ]; then
      echo $msg ERROR no paired backup node has been defined for NODE_TAG $NODE_TAG do so in base/local.bash::local-backup-tag
   else
	  local target_var=$(scm-backup-target $target_tag)
	  local remote=$target_var/scm/backup 
	  local source=$SCM_FOLD/backup/$LOCAL_NODE
 
      ssh $target_tag "mkdir -p  $remote"
      
      echo $msg transfer $source to $target_tag:$remote/ 
      local cmd1="rsync --delete-after -razvt $source $target_tag:$remote/ "
      echo $msg $cmd1
      eval $cmd1

   fi 
}


scm-backup-sudouser(){

   local msg="=== $FUNCNAME :"
   
   local user=$(apache-user)
   [ -z $user ] && echo $msg ERROR apache-user not defined && return 1 

   local sudouser
   if [ "$SUDO" == "" ]; then
      sudouser=""
    else
      sudouser="$SUDO -u $user"
    fi
    echo $sudouser
}


scm-recover-repo(){

   local msg="=== $FUNCNAME :"
   local name=${1:-dummy}   ## name of the backup
   local path=${2:-dummy}   ## absolute path to backup folder containing the last link  
   local dest=${3:-dummy}   ## destination folder, usually $SCM_FOLD/repos OR $SCM_FOLD/tracs 
   
   [ "$name" == "dummy" ] && echo $msg ERROR the name must be given && return 1 
   [ ! -d "$path" ]       && echo $msg ERROR path $path does not exist && return 1 
   [ ! -d "$dest" ]       && echo $msg ERROR destination folder $dest does not exist && return 1 
   
   local sudouser=$(scm-backup-sudouser)
   
   #
   #  hmm must be careful with copies of the tarballs to prevent collapsing the link
   #
   
   cd $path
   local stamp=$(readlink last)
   local target_fold=$path/$stamp
   cd $target_fold
   
   if [ "$?" == "1" ]; then
      echo $msg error target_fold $target_fold not found 
   else
   
      declare -a tgzs
      tgzs=($(ls -1 *.tar.gz))
      local ntgz=${#tgzs[@]}

      if [ "$ntgz" == "1" ]; then
      
         local tgz=${tgzs[0]} 
         local tgzname=${tgz%.tar.gz}
         local tgzpath=$target_fold/$tgzname.tar.gz
      
         cd $dest 
         
         if [ -d "$name" ]; then
            echo $msg the repository/folder:$name is present already in dest $dest , must delete the $name folder before can recover 
            echo $msg stamp $stamp target_fold $target_fold ==== tgz $tgz ===== tgzname $tgzname
         else
             
            echo $msg recovering repository/folder $name from tarball $tgzpath $tgzname into $(pwd) sudouser:[$sudouser] SUDO:[$SUDO]
            $SUDO cp $tgzpath .
            $SUDO tar zxvf $tgzname.tar.gz
            
            ## document the recovery via a link to the backup tarball
            $SUDO ln -s $tgzpath ${name}-scm-recover-repo
            
            $SUDO rm -f $tgzname.tar.gz
            
    
            ## svn tarballs have the revision number appended to their names
            if [ "$tgzname" != "$name" ]; then
              $SUDO mv $tgzname $name
            fi
            
            local user=$(apache-user)
            $SUDO chown -R $user $tgzname 
            
         fi      
      else
         echo $msg  ERROR there is not 1 tgz in target_fold $target_fold
      fi 
   fi
}






scm-backup-repo(){

   local msg="=== $FUNCNAME :" 
   local name=${1:-dummy}   ## name of the repo
   local path=${2:-dummy}   ## absolute path to the repo  
   local base=${3:-dummy}   ## backup folder
   local stamp=${4:-dummy}  ## date stamp
   
   echo $msg name $name path $path base $base stamp $stamp ===
   
   [ "$name" == "dummy" ]  &&  echo $msg ERROR the name must be given && return 1 
   [ ! -d "$path" ]        &&  echo $msg ERROR path $path does not exist && return 1 
   [ "$base" == "dummy" ]  &&  echo $msg ERROR the base must be given && return 1 
   [ "$stamp" == "dummy" ] &&  echo $msg ERROR the stamp must be given && return 1 
   
   local target_fold=$base/repos/$name/$stamp
   #   
   #  
   # hot-copy.py creates tgzs like : 
   #       name-rev.tar.gz 
   #       name-rev-index.tar.gz       index:1,2,3,...
   # 
   #  inside $target_fold , which must exist
   # 
     
   local hot_backup=$(svn-hotbackuppath)      
   [ ! -x $hot_backup ] && echo $msg ABORT no hot_backup script $hot_backup && return 1
                  			  	  
   local cmd="mkdir -p $target_fold &&  $hot_backup --archive-type=gz $path $target_fold && cd $base/repos/$name && rm -f last && ln -s $stamp last "   
   echo $msg $cmd
   eval $cmd
   
   
   # to check a integrity of a backed up repository , after unpacking 
   # svn co file:///tmp/hottest-6
}

scm-backup-trac(){

   local msg="=== $FUNCNAME :" 
   local name=${1:-dummy}     ## name of the trac
   local path=${2:-dummy}     ## absolute path to the trac
   local base=${3:-dummy}     ## backup folder
   local stamp=${4:-dummy}  ## date stamp
   
   echo $msg name $name path $path base $base stamp $stamp ===
   
   #
   #  perhaps the stamp should be above the name, and have only one stamp 
   #
   
   [ "$name" == "dummy" ] &&  echo the name must be given && return 1 
   [ ! -d "$path" ]       &&  echo ERROR path $path does not exist && return 1 
   [ "$base" == "dummy" ] &&  echo the base must be given && return 1 
   [ "$stamp" == "dummy" ] &&  echo the stamp must be given && return 1 
   
   
   local source_fold=$path
   local target_fold=$base/tracs/$name/$stamp/$name
   local parent_fold=$(dirname $target_fold)

   
   ## target_fold must NOT exist , but its parent should
   ## too many pythons around to rely on an external PYTHON_HOME
   
   local trac_admin
   if [ -x "/usr/local/bin/trac-admin" ]; then
      trac_admin=/usr/local/bin/trac-admin
   else 
	  trac_admin=$REFERENCE_PYTHON_HOME/bin/trac-admin
   fi	  
   
   [ ! -x $trac_admin ] && echo $msg ABORT no trac_admin at $trac_admin && return 1
   
   local cmd="mkdir -p $parent_fold && $trac_admin $source_fold hotcopy $target_fold && cd $parent_fold && tar -zcvf $name.tar.gz $name/* && rm -rf $name && cd $base/tracs/$name && rm -f last && ln -s $stamp last "
   echo $msg $cmd
   eval $cmd 
   
   #
   #  to check integrity of the sqlite database that is the heart of trac
   #   sqlite3 /path/to/env/db/trac.db
   #    > .help
   #    > .tables
   #    > .schema wiki
   #    > .dump            dumps the database as SQL statements 
   # 
}


scm-backup-folder(){

   local msg="=== $FUNCNAME :"
      
   local name=${1:-dummy}     ## 
   local path=${2:-dummy}     ## absolute path of folder to be backed up 
   local base=${3:-dummy}     ## backup folder
   local stamp=${4:-dummy}    ## date stamp
   
   echo $msg name $name path $path base $base stamp $stamp ===
   
   [ "$name" == "dummy" ] &&  echo the name must be given && return 1 
   [ ! -d "$path" ]       &&  echo ERROR path $path does not exist && return 1 
   [ "$base" == "dummy" ] &&  echo the base must be given && return 1 
   [ "$stamp" == "dummy" ] &&  echo the stamp must be given && return 1 
   
   local source_fold=$path
   local target_fold=$base/folders/$name/$stamp
   
   local cmd="mkdir -p $target_fold ; cd $(dirname $source_fold) ; rm -f $name.tar.gz ; tar -zcvf $name.tar.gz $(basename $source_fold)  ; cp $name.tar.gz $target_fold/ && cd $base/folders/$name && rm -f last && ln -s $stamp last "
   echo $msg "$cmd"
   eval $cmd
 
}






