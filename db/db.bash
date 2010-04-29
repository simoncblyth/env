
db-source(){ echo ${BASH_SOURCE:-$(env-home)/db/db.bash} ; }
db-srcdir(){ echo $(dirname $(db-source)); }
db-cd(){ cd $(db-srcdir) ; }
db-vi(){ vi $(db-source) ; }
db-up(){ source $(db-source) ; }
db-env(){ echo -n ; }


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
           performs backups and purges 


     db-backup-rsync-monitor-
            emit to stdout the listing of todays and yesterdays .sql.gz
     db-backup-rsync-monitor
            send monitoring email 

     db-backup-recover <dbname>
           recovers the gzipped mysqldump into local database 
           with name based on the day eg: <dbname>_20100101 

     db-test
          runs nosetests from  $(db-srcdir) 
          one of the tests compares the table counts in 
          dybdb1 with those in the recovered copy 
      
     Checking the mysqldump ... tiz all within db context 
         gunzip -c $(db-backup-rsync-sqz) | cat - | more


    This is in use with rsync transfers  :
        dybdb1.ihep.ac.cn > cms01
        dybdb2.ihep.ac.cn > cms01 


EOU
}

db-backup-basedir(){ echo /var/dbbackup ; }
db-backup-rsyncdir(){ echo $(db-backup-basedir)/rsync ; }
db-backup-hostdir(){ echo $(db-backup-basedir)/$(hostname) ; }
db-backup-daydir(){  echo $(db-backup-hostdir $*)/$(date +"%Y%m%d") ; } 
db-backup-rdaydir(){  echo $(db-backup-rsyncdir)/${1:-nohost}/$(db-today) ; } 
db-backup-ryaydir(){  echo $(db-backup-rsyncdir)/${1:-nohost}/$(db-yesterday) ; } 
db-backup-names(){   echo testdb offline_db ; }
db-backup-keep(){    echo 7 ; }

db-today(){ date +"%Y%m%d" ; }
db-yesterday(){
   local fmt=${1:-%Y%m%d}
   case $(uname) in 
      Linux) date -d yesterday +$fmt ;;
     Darwin) date -r $(( $(date +%s) - 86400 ))  +$fmt ;;
   esac
}


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


db-backup-rsync-sqz(){
   local name=${1:-testdb}
   local host=${3:-dybdb1.ihep.ac.cn}
   echo $(db-backup-rdaydir $host)/$name.sql.gz
}

db-backup-rsync-monitor-(){
   local msg="=== $FUNCNAME :"
   local rdir=$(db-backup-rsyncdir) 
   echo $msg $(hostname) cf todays and yesterdays  .sql.gz 
   echo $msg $(date)
   local host
   local name
   for name in $(db-backup-names) ; do 
        echo $msg $name 
        ls -1 $rdir | while read host ; do
          local rdaydir=$(db-backup-rdaydir $host)
          local ryaydir=$(db-backup-ryaydir $host)
          local sgz="$rdaydir/$name.sql.gz"
          local ygz="$ryaydir/$name.sql.gz"
          if [ -f "$sgz" ]; then
              ls -l $ygz
              ls -l $sgz
          else
              echo $msg ERROR missing $sgz  
          fi
       done 
   done 
}

db-backup-rsync-monitor(){
   local tmp=/tmp/env/${FUNCNAME}.txt && mkdir -p $(dirname $tmp)
   $FUNCNAME- > $tmp 2>&1   
   python-
   python-sendmail $tmp 
}


db-cautious(){  
   ## dont use defaults as potential to wipe your DB 
   mysql --no-defaults --host=localhost --user=root -p $1 
}

db-recover(){
  [ "$(hostname)" != "cms01.phys.ntu.edu.tw" ] && echo sorry too dangerous ... && return 1
  private-
  mysql --host=localhost --user=$(private-val RECOVER_USER) --password=$(private-val RECOVER_PASSWORD) $1
}

db-name-today(){     echo ${1}_$(db-today) ; }
db-name-yesterday(){ echo ${1}_$(db-yesterday) ; }

db-grant(){ 
  local name=${1:-testdb}
  local rname=${2:-WEBSRV}
  private-
  local ruser=$(private-val ${rname}_USER)
  local rhost=$(private-val ${rname}_HOST)
  local rpass=$(private-val ${rname}_PASS)
  cat << EOG
grant select on $(db-name-today $name).* to $ruser@$rhost identified by '$rpass' ;
EOG
}

db-backup-recover(){
   local msg="=== $FUNCNAME :"
   local name=${1:-testdb}
   local sqz=$(db-backup-rsync-sqz $name)
   [ ! -f "$sqz" ] && echo $msg ABORT sqz $sqz does not exist && return 2
   local dbtoday=$(db-name-today $name)
   local dbyesterday=$(db-name-yesterday $name)
   echo $msg name $name sqz $sqz dbtoday $dbtoday dbyesterday $dbyesterday
   local rc
   ## stream direct from the sqz into the DB as the sqz is owned by the unprivileged rsync only user 
   echo "create database if not exists $dbtoday ;" | db-recover 
   gunzip -c $sqz                                  | db-recover $dbtoday
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT error $rc && return $rc   
   
   ! db-exists $dbtoday    && echo $msg FAILED to create DB $dbtoday  && return 3
   echo $msg SUCCEEDED to create DB $dbtoday
   db-grant $name WEBSRV | db-recover 
   
   echo $msg dropping $dbyesterday
   echo "drop   database if exists $dbyesterday ;" | db-recover 
   db-exists $dbyesterday && echo $msg FAILED to drop DB $dbyesterday  && return 3
}

db-exists(){
  [ "$(echo 'show databases ;' | db-recover | grep $1 )" == "$1" ] && return 0 || return 1
}

db-test-(){
   db-cd
   nosetests -v
}

db-test(){
   local msg="=== $FUNCNAME :"
   local rc 
   local tmp=$(db-logdir)/${FUNCNAME}.txt && mkdir -p $(dirname $tmp)
   echo $msg writing to $tmp
   db-test- > $tmp 2>&1   
   rc=$?
   python-
   [ "$?" != "0" ] && echo $msg TEST FAILURES from $PWD && python-sendmail $tmp && return $rc
   #echo $msg TEST SUCCEEDED && rm -f $tmp
}

db-logdir(){  echo $HOME/log/env ; }
db-logs(){
   cd $(db-logdir)
   echo $PWD $(date)
   ls -l 
}
