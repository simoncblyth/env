
db-source(){ echo ${BASH_SOURCE:-$(env-home)/db/db.bash} ; }
db-srcdir(){ echo $(dirname $(db-source)); }
db-cd(){ cd $(db-srcdir) ; }
db-vi(){ vi $(db-source) ; }
db-up(){ source $(db-source) ; }
db-env(){ echo -n ; }


db-usage(){
  cat << EOU

    killed the sync as very slow network at Daya Bay after transfering 

        g4pb:rsync blyth$ db-;db-backup-rsync-from-c
        dybdb1.ihep.ac.cn/20101227/offline_db.sql.gz
        dybdb1.ihep.ac.cn/20101227/testdb.sql.gz
        ^CKilled by signal 2.

    recover into G manually 
   
        g4pb:rsync blyth$ db- ; db-backup-recover-sqz dybdb1.ihep.ac.cn/20101227/offline_db.sql.gz
        1229.28 real        35.67 user         3.16 sys
   

------------
     db-backup-purge
          delete local day folders beneath $(db-backup-hostdir)
          retaining only the last $(db-backup-keep) days
          ... relies on sort order of the names 

     db-backup <dbname>
           mysqldump creating <dbname>.sql.gz in db-backup-daydir
           the password/user/host defaults are obtained from
            /etc/my.cnf avoiding keeping config info in this script

     db-backup-names : $(db-backup-names)
           names of databases to be backed up and recovered

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

     db-backup-recover <dbname> <host>
           recovers the gzipped mysqldump into local database 
           with name based on the day eg: <dbname>_20100101 
           from the mysqldump rsynced from host (eg dybdb1.ihep.ac.cn) 


           ISSUE ... 
             works in cron job but fails when running interactively fails with 
                 ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/tmp/mysql.sock' (2)

             connection can be made to work by changing db-recover to specifiy the RECOVER_HOST rather than
             using localhost ?



     db-test
          runs nosetests from  $(db-srcdir) 
          one of the tests compares the table counts in 
          dybdb1 with those in the recovered copy 
      
     Checking the mysqldump ... tiz all within db context 
         gunzip -c $(db-backup-rsync-sqz) | cat - | more



     db-backup-rsync-sqz  <testdb> <dybdb1.ihep.ac.cn>
          Absolute path to the mysqldump file         

         -rw-r--r--  1 dayabayscp dayabayscp 1426071 May  3 16:55 /var/dbbackup/rsync/dybdb1.ihep.ac.cn/20100503/testdb.sql.gz
         -rw-r--r--  1 dayabayscp dayabayscp 1426066 May  3 04:08 /var/dbbackup/rsync/dybdb2.ihep.ac.cn/20100503/testdb.sql.gz

            hmm suspect a double-cron of the dybdb1 dump 

         -rw-r--r--  1 dayabayscp dayabayscp 1426070 May  4 04:10 /var/dbbackup/rsync/dybdb1.ihep.ac.cn/20100504/testdb.sql.gz
         -rw-r--r--  1 dayabayscp dayabayscp 1426065 May  4 04:10 /var/dbbackup/rsync/dybdb2.ihep.ac.cn/20100504/testdb.sql.gz



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

db-today(){ [ -n "$DB_DATE" ] && echo $_DATE || date +"%Y%m%d" ; }
db-yesterday(){
   local fmt=${1:-%Y%m%d}
   case $(uname) in 
      Linux) date -d yesterday +$fmt ;;
     Darwin) date -r $(( $(date +%s) - 86400 ))  +$fmt ;;
   esac
}

db-standalone(){
   local msg="=== $FUNCNANE : "
   local iwd=$PWD
   local dir=/tmp/$FUNCNAME &&  mkdir -p $dir && cd $dir
   local url=http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/DybPython/python/DybPython
   
   echo $msg mysql-python is required 
   python -c "import MySQLdb as _ ; print _.__version__ " 
   [ ! -f "$dir/db.py" ]     && svn export $url/db.py
   [ ! -f "$dir/dbconf.py" ] && svn export $url/dbconf.py

   python $dir/db.py $*
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
  time mysqldump  $name > $name.sql && gzip $name.sql 
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

db-backup-rsync-from-c(){
   [ "$NODE_TAG" == "C" ] && echo ABORT not allowed to do this on node C && return 1
   local tag=C
   local tgt=$(db-backup-rsyncdir)/    ## remote dir named after local hostname
   local cmd="rsync -e ssh --delete-after -razvt $tag:$tgt $tgt "
   echo $cmd
   eval $cmd
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
   local host=${2:-dybdb1.ihep.ac.cn}
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


db-recover-credentials-(){ private- ; cat << EOC
[client]
host      = $(private-val RECOVER_HOST)
database  = $1
user      = $(private-val RECOVER_USER)
password  = $(private-val RECOVER_PASSWORD)
EOC
}

db-recover-credentials(){
  ## rewrite credentials every time to avoid habing to put db name into path or some such 
  local path=~/.my.$FUNCNAME.cnf
  $FUNCNAME- $* > $path
  chmod go-rwx $path
  echo $path
}


db-recover(){
  private-
  local host=$(private-val RECOVER_HOST)
  case $host in 
     cms01.phys.ntu.edu.tw|localhost) echo -n                                    ;;
                                  *)  echo sorry too dangerous ... && return 1   ;;
  esac

  local cfg=$(echo $(db-recover-credentials $1))

#  ${DB_TIME} mysql --no-defaults --host=$host --user=$(private-val RECOVER_USER) --password=$(private-val RECOVER_PASSWORD) $1

  ${DB_TIME} mysql --defaults-file=$cfg 

  [ ${#cfg} -lt ${#HOME} -o ${#cfg} -lt 10 ] && echo $msg SANITY CHECK FAILURE FOR cfg $cfg && return 1
  [ -f "$cfg" ] && rm -f "$cfg"
}


db-name-today(){     echo ${1}_$(db-today) ; }
db-name-yesterday(){ echo ${1}_$(db-yesterday) ; }

db-user(){ private- ; private-val ${1:-DAYABAY}_USER ; }
db-host(){ private- ; private-val ${1:-DAYABAY}_HOST ; }
db-pass(){ private- ; private-val ${1:-DAYABAY}_PASS ; }





db-create-user-(){  cat << EOC
create user '$(db-user $1)'@'$(db-host $1)' IDENTIFIED BY '$(db-pass $1)' ; 
EOC
}
db-create-user(){
  [ "$(mysql --version)" == "mysql  Ver 14.7 Distrib 4.1.22, for redhat-linux-gnu (i686) using readline 4.3" ] && echo this mysql does not have create user ... just use db-grant && return 1  
  $FUNCNAME- | db-recover
}

db-grant-(){ 
   local pfx=${1:-DAYABAY}  ## private var prefix
   local db=${2:-offline_db}
   cat << EOC
grant select on $(db-name-today ${db:-offline_db}).* to '$(db-user $pfx)'@'$(db-host $pfx)' identified by '$(db-pass $pfx)' ;
EOC
}
db-grant(){
  $FUNCNAME-  | db-recover 
}


db-backup-recover-sqz(){
   local msg="=== $FUNCNAME :"
   local sqz=$1
   [ ! -f "$sqz" ] && echo $msg ABORT sqz $sqz does not exist && return 2
  
   local name=$(basename $sqz)
   name=${name/.*/}
   case $name in 
     offline_db) echo -n ;;
         testdb) echo -n ;;
              *) echo $msg unexpected basename  $name ... aborting && return 1 ;;
   esac

  local date=$(basename $(dirname $sqz))
  case $date in
     2010????) echo -n ;;
     2011????) echo -n ;;
            *) echo expecting a date && return 1 ;;
  esac

    
  local dbrecover=${name}_${date}
  echo $msg recovering from sqz $sqz into DB $dbrecover 

  echo "create database if not exists $dbrecover ;" | db-recover 
  gunzip -c $sqz                                  | DB_TIME=time  db-recover $dbrecover
  

}


db-backup-recover(){
   local msg="=== $FUNCNAME :"
   local name=${1:-offline_db}
   local host=${2:-dybdb1.ihep.ac.cn}

   local sqz=$(db-backup-rsync-sqz $name $host)
   [ ! -f "$sqz" ] && echo $msg ABORT sqz $sqz does not exist && return 2

   local dbtoday=$(db-name-today $name)
   local dbyesterday=$(db-name-yesterday $name)
   echo $msg name $name sqz $sqz dbtoday $dbtoday dbyesterday $dbyesterday
   local rc
   ## stream direct from the sqz into the DB as the sqz is owned by the unprivileged rsync only user 
   echo "create database if not exists $dbtoday ;" | db-recover 
   gunzip -c $sqz                                  | DB_TIME=time db-recover $dbtoday
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT error $rc && return $rc   
   
   ! db-exists $dbtoday    && echo $msg FAILED to create DB $dbtoday  && return 3
   echo $msg SUCCEEDED to create DB $dbtoday
   db-grant $name DAYABAY | db-recover 
   
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

db-logdir(){  echo $HOME/cronlog ; }
db-logs(){
   cd $(db-logdir)
   echo $PWD $(date)
   ls -l 
}


db-status(){
   echo status | $(my.py offline_db1)
   echo status | $(my.py offline_db2)
}

