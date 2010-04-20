
db-source(){ echo ${BASH_SOURCE:-$(env-home)/db/db.bash} ; }
db-vi(){ vi $(db-source) ; }
db-up(){ source $(db-source) ; }
db-env(){ echo -n ; }

pymysql-(){      . $ENV_HOME/db/pymysql.bash && pymysql-env $* ; }
pysqlite-(){     . $ENV_HOME/db/pysqlite.bash  && pysqlite-env  $* ; }


db-usage(){
  cat << EOU

     db-backup-purge
          delete local day folders beneath $(db-backup-hostdir)
          retaining only the last $(db-backup-keep) days
          ... relies on sort order of the names 

     db-backup <dbname>
           backup db using mysqldump
           the password/user/host defaults are obtained from
            /etc/my.cnf avoiding keeping config info in this script

     db-backup-names : $(db-backup-names)

     db-backup-rsync target.node  
          rsync local dir $(db-backup-hostdir)
          to remote target.node into dir $(db-backup-rsyncdir)/ 
          using the local hostname for identification 

     db-backup-daily
           perfomrms backups and purges 

    This is in use with rsync transfers  :
        dybdb1.ihep.ac.cn > cms01
        dybdb2.ihep.ac.cn > cms01 


EOU
}

db-backup-basedir(){ echo /var/dbbackup ; }
db-backup-rsyncdir(){ echo $(db-backup-basedir)/rsync ; }
db-backup-hostdir(){ echo $(db-backup-basedir)/$(hostname) ; }
db-backup-daydir(){  echo $(db-backup-hostdir)/$(date +"%Y%m%d") ; } 
db-backup-names(){   echo testdb offline_db ; }
db-backup-keep(){    echo 7 ; }

db-backup-cd(){ cd $(db-backup-daydir) ; }
db-backup-ls(){ ls -Ralst $(db-backup-basedir) ; }

db-backup-daily(){
  local name
  for name in $(db-backup-names) ; do 
     db-backup $name
  done
  db-backup-purge
}



db-backup(){
  local msg="=== $FUNCNAME : "
  local name=${1:-testdb}
  local dir=$(db-backup-daydir)
  local iwd=$PWD
  mkdir -p $dir && cd $dir
  ## password/user/host defaults from /etc/my.cnf
  echo $msg mysqldump of $name from $PWD
  local sgz="$name.sql.gz"
  [ -f "$sgz" ] && rm -f "$sgz" 
  mysqldump  $name > $name.sql && gzip $name.sql 
  cd $iwd
}


db-backup-purge(){

  local msg="=== $FUNCNAME : "
  local nmax=$(db-backup-keep)
  local days
  local iday
  local nday 
  
  declare -a days

  cd $(db-backup-hostdir)
     
  days=($(find . -name '????????' | sort ))
  nday=${#days[@]}
     
  echo $msg from $PWD  nday:$nday nmax:$nmax
  iday=0
  while [ "$iday" -lt "$nday" ]
  do    
    local day=${days[$iday]}
    [ ! ${#day} -gt 8 ] && echo $msg ERROR day \"$day\" is not gt 8 chars long && return 1 

    if [ $(( $nday - $iday > $nmax )) == 1 ]; then 
        local cmd="rm -rf $day"
        echo delete $day ... $cmd
        eval $cmd 
    else
        echo retain $day
    fi 
    let "iday = $iday + 1"
  done
     
}


db-backup-rsync(){

   local msg="# === $FUNCNAME : "
   local tag=$1
   [ "$tag" == "" ] && echo $msg ABORT must enter target node name or ssh tag && return 1
   [ "$tag" == "$(hostname)" ] && echo $msg ABORT cannot rsync to self && return 1

   local src=$(db-backup-hostdir)      ##local 
   local tgt=$(db-backup-rsyncdir)/    ## remote dir named after local hostname

   local pmd="ssh $tag \"mkdir -p $tgt\" "
   echo $pmd
   eval $pmd    

   local cmd="rsync -e ssh --delete-after -razvt $src $tag:$tgt "
   echo $cmd
   eval $cmd
}

