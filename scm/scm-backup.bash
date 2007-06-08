

scm-backup-all(){

   ## first backup the 
   
   local stamp=$(base-datestamp now %Y/%m/%d/%H%M%S)
   local base=$SCM_FOLD/backup/$stamp
   
   for path in $SCM_FOLD/repos/*
   do   
       local name=$(basename $path)
       scm-backup-repo $name $path $base         
   done
   
   for path in $SCM_FOLD/tracs/*
   do   
       local name=$(basename $path)
       scm-backup-trac $name $path $base         
   done
}



scm-backup-repo(){

   local name=${1:-dummy}
   local path=${2:-dummy}
   local base=${3:-dummy}
   
   [ "$name" == "dummy" ] && ( echo the name must be given && return 1 )
   [ -d "$path" ] || ( echo ERROR path $path does not exist && return 1 )
   [ "$base" == "dummy" ] && ( echo the base must be given && return 1 )
   
   local target_fold=$base/repos/$name
   #   
   #  
   # hot-copy.py creates tgzs like : 
   #       name-rev.tar.gz 
   #       name-rev-index.tar.gz       index:1,2,3,...
   # 
   #  inside $target_fold , which must exist
   # 
           
   local cmd="mkdir -p $target_fold &&  $LOCAL_BASE/svn/build/subversion-1.4.0/tools/backup/hot-backup.py --archive-type=gz $path $target_fold "   
   echo $cmd
   eval $cmd
   
}

scm-backup-trac(){

   local name=${1:-dummy}
   local path=${2:-dummy}
   local base=${3:-dummy}
   
   #
   #  perhaps the stamp should be above the name, and have only one stamp 
   #
   
   [ "$name" == "dummy" ] && ( echo the name must be given && return 1 )
   [ -d "$path" ] || ( echo ERROR path $path does not exist && return 1 )
    [ "$base" == "dummy" ] && ( echo the base must be given && return 1 )
   
   local source_fold=$path
   local target_fold=$base/tracs/$name
   local parent_fold=$(dirname $target_fold)
   
   ## target_fold must NOT exist , but its parent should
   
   local cmd="mkdir -p $parent_fold && $PYTHON_HOME/bin/trac-admin $source_fold hotcopy $target_fold"
   echo $cmd
   eval $cmd 
   
}

