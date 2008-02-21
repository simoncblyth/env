
# invoke with :
#     sudo bash -lc scm-backup-all
#
#   need to do this in roots crontab ,  do as a script ?
#
#

scm-backup-all(){
   
   local stamp=$(base-datestamp now %Y/%m/%d/%H%M%S)
   local base=$SCM_FOLD/backup/$LOCAL_NODE
   
   for path in $SCM_FOLD/repos/*
   do   
       local name=$(basename $path)
       scm-backup-repo $name $path $base $stamp        
   done
   
   for path in $SCM_FOLD/tracs/*
   do   
       local name=$(basename $path)
       scm-backup-trac $name $path $base $stamp        
   done
   
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
      
      for path in $base/*
      do   
          local name=$(basename $path)
          scm-recover-repo $name $path $dest   
		  #
		  #  eg:
		  #    name : "workflow" 
		  #    path : /var/scm/backup/g4pb/tracs/workflow   or repos equivalent    
          #    dest : /var/scm/tracs or /var/scm/repos 
	  done
      
   done 

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

  for path in $SCM_FOLD/backup/$node/{tracs,repos}/* 
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




scm-backup-rsync(){

   # 
   # rsync the local backup repository to an off box mirror on the paired $BACKUP_TAG node 
   #   - have to set up ssh keys to allow non-interactive sshing 
   # 
   #  hmm the passwordless ssh is not setup for "root" user , so have to do this as me, but the above backup as root
   #

   local target_tag=${1:-$BACKUP_TAG}
   
   if [ "$target_tag" != "$BACKUP_TAG" ]; then 
      local vname=VAR_BASE_$target_tag 
      eval _VAR_BASE_BACKUP=\$$vname
      target_var=${_VAR_BASE_BACKUP:-$VAR_BASE_U}
      echo ======== scm-backup-rsync to non-default target_tag:$target_tag  target_var:$target_var
   else
      target_var=$VAR_BASE_BACKUP
   fi

   if [ "X$target_tag" == "X" ]; then
      echo no paired backup node has been defined for node $LOCAL_NODE
   else

      local source=$SCM_FOLD/backup/$LOCAL_NODE
      local remote=$target_var/scm/backup 
 
      ssh $target_tag "mkdir -p  $remote"
      
      echo ============== transfer $source to $target_tag:$remote/ 
      local cmd1="rsync --delete-after -razvt $source $target_tag:$remote/ "
      echo $cmd1
      eval $cmd1

   fi 
}


scm-recover-repo(){

   local name=${1:-dummy}   ## name of the repo
   local path=${2:-dummy}   ## absolute path to the repo  
   local dest=${3:-dummy}   ## destination folder, usually $SCM_FOLD/repos OR $SCM_FOLD/tracs 
   
   [ "$name" == "dummy" ] && ( echo the name must be given && return 1 )
   [ -d "$path" ] || ( echo ERROR path $path does not exist && return 1 )
   [ -d "$dest" ] || ( echo ERROR destination folder $dest does not exist && return 1 )
   
   
   #
   #  hmm must be careful with copies of the tarballs to prevent collapsing the link
   #
   
   cd $path
   local stamp=$(readlink last)
   local target_fold=$path/$stamp
   cd $target_fold
   
   if [ "$?" == "1" ]; then
      echo error target_fold $target_fold not found 
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
            echo === scm-recover-repo ===  the repository:$name is present already , must delete this before can recover 
            echo stamp $stamp target_fold $target_fold ==== tgz $tgz ===== tgzname $tgzname
         else
             
            echo === scm-recover-repo === recovering repository $name from tarball $tgzpath $tgzname into $(pwd)
            sudo -u $APACHE2_USER cp $tgzpath .
            sudo -u $APACHE2_USER tar zxvf $tgzname.tar.gz
            
            ## document the recovery via a link to the backup tarball
            sudo -u $APACHE2_USER ln -s $tgzpath ${name}-scm-recover-repo
            sudo rm -f $tgzname.tar.gz
            
            ## svn tarballs have the revision number appended to their names
            if [ "$tgzname" != "$name" ]; then
              sudo -u $APACHE2_USER mv $tgzname $name
            fi
            
         fi      
      else
         echo scm-recover-repo ERROR there is not 1 tgz in target_fold $target_fold
      fi 
   fi
}





scm-backup-repo(){

   local name=${1:-dummy}   ## name of the repo
   local path=${2:-dummy}   ## absolute path to the repo  
   local base=${3:-dummy}   ## backup folder
   local stamp=${4:-dummy}  ## date stamp
   
   [ "$name" == "dummy" ] && ( echo the name must be given && return 1 )
   [ -d "$path" ] || ( echo ERROR path $path does not exist && return 1 )
   [ "$base" == "dummy" ] && ( echo the base must be given && return 1 )
   [ "$stamp" == "dummy" ] && ( echo the stamp must be given && return 1 )
   
   local target_fold=$base/repos/$name/$stamp
   #   
   #  
   # hot-copy.py creates tgzs like : 
   #       name-rev.tar.gz 
   #       name-rev-index.tar.gz       index:1,2,3,...
   # 
   #  inside $target_fold , which must exist
   # 
       
   local hot_backup		     
   if [ "$NODE_APPROACH" == "stock" ]; then
	 hot_backup=$LOCAL_BASE/svn/tools/backup/hot-backup.py
   else
	 hot_backup=$LOCAL_BASE/svn/build/subversion-1.4.0/tools/backup/hot-backup.py	
   fi		
			  	  
   local cmd="mkdir -p $target_fold &&  $hot_backup --archive-type=gz $path $target_fold && cd $base/repos/$name && rm -f last && ln -s $stamp last "   
   echo $cmd
   eval $cmd
   
   
   # to check a integrity of a backed up repository , after unpacking 
   # svn co file:///tmp/hottest-6
}

scm-backup-trac(){

   local name=${1:-dummy}     ## name of the trac
   local path=${2:-dummy}     ## absolute path to the trac
   local base=${3:-dummy}     ## backup folder
   local stamp=${4:-dummy}  ## date stamp
   
   #
   #  perhaps the stamp should be above the name, and have only one stamp 
   #
   
   [ "$name" == "dummy" ] && ( echo the name must be given && return 1 )
   [ -d "$path" ] || ( echo ERROR path $path does not exist && return 1 )
   [ "$base" == "dummy" ] && ( echo the base must be given && return 1 )
   [ "$stamp" == "dummy" ] && ( echo the stamp must be given && return 1 )
   
   
   local source_fold=$path
   local target_fold=$base/tracs/$name/$stamp/$name
   local parent_fold=$(dirname $target_fold)
   local 
   
   ## target_fold must NOT exist , but its parent should
   ## too many pythons around to rely on an external PYTHON_HOME
   
   local trac_admin
   if [ "$NODE_APPROACH" == "stock" ]; then
      trac_admin=/usr/local/bin/trac-admin
   else 
	  trac_admin=$REFERENCE_PYTHON_HOME/bin/trac-admin
   fi	  
   
   local cmd="mkdir -p $parent_fold && $trac_admin $source_fold hotcopy $target_fold && cd $parent_fold && tar -zcvf $name.tar.gz $name/* && rm -rf $name && cd $base/tracs/$name && rm -f last && ln -s $stamp last "
   echo $cmd
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

