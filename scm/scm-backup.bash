
## invoke with :
##     sudo bash -lc scm-backup-all

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
}


scm-backup-rsync(){

   # 
   # rsync the local backup repository to an off box mirror on the paired $BACKUP_TAG node 
   #   - have to set up ssh keys to allow non-interactive sshing 
   # 

   if [ "X$BACKUP_TAG" == "X" ]; then
      echo no paired backup node has been defined for node $LOCAL_NODE
   else

      local source=$SCM_FOLD/backup/$LOCAL_NODE
      local remote=$VAR_BASE_BACKUP/scm/backup 
 
      ssh $BACKUP_TAG "mkdir -p  $remote"
      rsync -razvt $source $BACKUP_TAG:$remote/ 

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
           
   local cmd="mkdir -p $target_fold &&  $LOCAL_BASE/svn/build/subversion-1.4.0/tools/backup/hot-backup.py --archive-type=gz $path $target_fold && cd $base/repos/$name && rm -f last && ln -s $stamp last "   
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
   
   local cmd="mkdir -p $parent_fold && $PYTHON_HOME/bin/trac-admin $source_fold hotcopy $target_fold && cd $parent_fold && tar -zcvf $name.tar.gz $name/* && rm -rf $name && cd $base/tracs/$name && rm -f last && ln -s $stamp last "
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

