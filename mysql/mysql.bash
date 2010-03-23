mysql-src(){    echo mysql/mysql.bash ; }
mysql-source(){ echo ${BASH_SOURCE:-$(env-home)/$(mysql-src)} ; }
mysql-vi(){     vim $(mysql-source) ; }
mysql-usage(){
  cat << EOU

    Traditional redhat control :

       sudo /sbin/service mysqld start
       sudo /sbin/service mysqld status
       sudo /sbin/service mysqld stop

    mysql-sv
        Add to supervisor control ...
        needs to be configured to run as root


    mysql-triplet-edit "mysqld|log-bin|binlog"



    mysql-python-*

          normally can just 
              easy_install mysql-python
              pip install mysql-python

          but on WW with multiple mysql and python, suspect 
          confusion of these auto builds... with crossed include
          and lib dirs 

          it appears the mysql_config in the PATH dictates the 
          mysql that is built against but ...



    mysql-showdatabase

    mysql-dumpall

          CAUTION AS EACH db is dumped separately it is possible that the tables in i
          different DB will be inconsistent if one were to operate in a writing related tables to separate DB manner ?
  




EOU
}

mysql-env(){
  local msg="=== $FUNCNAME :"
  local bindir=$(mysql-bindir)
  if [ -n "$bindir" ]; then 
     ##env-prepend $bindir   a prexisting lowere level bindir was preventing this being brought to the front 
     export PATH=$bindir:$PATH
  else
     [ -n "$MYSQL_DBG" ] && echo $msg no mysql-bindir not defined 
  fi 

  local libdir=$(mysql-libdir)
  if [ -n "$libdir" ]; then 
     env-llp-prepend $libdir
  else
     [ -n "$MYSQL_DBG" ] && echo $msg no mysql-libdir not defined 
  fi 
}

mysql-bindir-(){
  case $NODE_TAG in 
    WW) echo /usr/local/mysql/bin ;;
  esac 
}

mysql-bindir(){
   pkgr-
  case $(pkgr-cmd) in
    port) echo /opt/local/lib/mysql5/bin ;;
       *) mysql-bindir- ;;
  esac
}

mysql-libdir(){
  case $NODE_TAG in 
     WW) echo /usr/local/mysql/lib/mysql ;;
     WW_other) echo /usr/lib/mysql ;;
  esac
}




mysql-versions(){
   echo "select version() ; " | mysql-sh
   mysql_config --version
   python -c "import MySQLdb as _ ; print 'MySQLdb:%s' % _.__version__ "
}

mysql-create-db(){
   local msg="=== $FUNCNAME :"
   private-
   local dbname=$(private-val DATABASE_NAME)
   [ "$dbname" == "" ] && echo $msg ABORT the DATABASE_NAME value is not defined && return 1
   echo "create database if not exists $dbname ;"  | mysql-sh- 
   mysql-show-tables
}

mysql-show-tables(){ echo "show tables ;" | mysql-sh ; }


mysql-five(){
   [ "$(uname)" == "Darwin" ] && echo 5 ; 
}

mysql-sh-(){
   private-
   [ -n "$MYSQL_DBG" ] && echo mysql$(mysql-five) --host=$(private-val DATABASE_HOST) --user=$(private-val DATABASE_USER) --password=$(private-val DATABASE_PASSWORD) $1
   mysql$(mysql-five) --host=$(private-val DATABASE_HOST) --user=$(private-val DATABASE_USER) --password=$(private-val DATABASE_PASSWORD) $1
}

mysql-sh(){
   private-
   mysql-sh- $(private-val DATABASE_NAME) ; 
}

mysql-showdatabases(){
   echo show databases | mysql-sh- --skip-column-names
}

mysql-bkpdir(){
   local base=${1:-/tmp/env/$FUNCNAME}
   local day=${2:-$(date +"%Y%m%d")}
   echo $base/$day
}

mysql-dump(){
   private-
   mysqldump --host=$(private-val DATABASE_HOST) --user=$(private-val DATABASE_USER) --password=$(private-val DATABASE_PASSWORD) $1
}


mysql-dumpall(){

  local db_
   local cmd
   local iwd=$PWD
   local dir=$(mysql-bkpdir $*)
   mkdir -p $dir && cd $dir
   mysql-showdatabases | while read db_ ; do
       mysql-dump $db_ > $db_.sql
   done
   cd $iwd
}




mysql-cnf(){
  pkgr-
  case $(pkgr-cmd) in
    yum) echo /etc/my.cnf ;;
      *) echo /etc/my.cnf
  esac
}
mysql-edit(){ sudo vi $(mysql-cnf) ; }
mysql-triplet-edit(){
  ini-
  ini-triplet-edit $(mysql-cnf) $* 
}



mysql-ini(){
  pkgr-
  case $(pkgr-cmd) in
    yum) echo /etc/init.d/mysqld  ;;
   ipkg) echo /opt/etc/init.d/S70mysqld ;;
   port) echo /opt/local/share/mysql5/mysql/mysql.server ;; 
  esac
}
mysql-post(){
  case $(pkgr-cmd) in 
    port) echo 5 ;;
       *) echo -n ;;
  esac
}


mysql-init(){    sudo $(mysql-ini) $* ; }

mysql-start(){   mysql-init start ; }
mysql-stop(){    mysql-init stop ; }
mysql-restart(){ mysql-init restart ; }
mysql-status(){  mysql-init status ; }

mysql-ps(){     ps aux | grep mysql ; }

mysql-admin(){
   private-
   mysqladmin$(mysql-post) -u $(private-val DATABASE_USER) --password=$(private-val DATABASE_PASSWORD) $*
}

mysql-logpath(){ echo /var/log/mysqld.log ; }
mysql-tail(){    sudo tail -f $(mysql-logpath) ; }

mysql-notes(){
  cat << EOU

    Installed 4.1.22 with yum $(env-wikiurl)/MySQL onto C2
        http://dev.mysql.com/doc/refman/4.1/en/
   This was prior to C2 being reinstalled 

   invenio requires some specific settings in my.cnf
       max_allowed_packet at least 4M
       default-character-set=utf8     (multiple places ?)

[blyth@cms02 ~]$ cat /etc/my.cnf
[mysqld]
datadir=/var/lib/mysql
socket=/var/lib/mysql/mysql.sock
user=mysql
# Default to using old password format for compatibility with mysql 3.x
# clients (those using the mysqlclient10 compatibility package).
old_passwords=1

[mysqld_safe]
err-log=/var/log/mysqld.log
pid-file=/var/run/mysqld/mysqld.pid


[blyth@cms02 ~]$ cat  /var/lib/mysql/my.cnf
cat: /var/lib/mysql/my.cnf: No such file or directory

[blyth@cms02 ~]$ l /var/lib/mysql/
total 20552
-rw-rw----  1 mysql mysql 10485760 Mar 27 14:10 ibdata1
-rw-rw----  1 mysql mysql  5242880 Mar 27 14:10 ib_logfile0
-rw-rw----  1 mysql mysql  5242880 Mar 27 14:10 ib_logfile1
drwx------  2 mysql mysql     4096 Mar 27 14:10 mysql
srwxrwxrwx  1 mysql mysql        0 Mar 27 14:10 mysql.sock
drwx------  2 mysql mysql     4096 Mar 27 14:10 test
[blyth@cms02 ~]$ 
EOU

}

mysql-tt-(){ cat << EOT
create table tt (name varchar(20) );
insert into tt (name) values ("simon") ; 
EOT
}


mysql-install-yum(){
   sudo yum install mysql-server
}

mysql-install-port(){
   sudo port install mysql5 +server
}


mysql-py-install(){
  if [ "$NODE_TAG" == "G" ] ; then
     easy_install MySQL-python
  else
     echo klop
  fi 


}


mysql-pidpath(){ echo /var/run/mysqld/mysqld.pid ; }
mysql-ps(){ ps aux | grep mysql ; }
mysql-sv-(){ 
   local msg="=== $FUNCNAME :"
   local pidproxy=$(which pidproxy)
   local pidpath=$(mysql-pidpath)
   local mysqld_safe=$(which mysqld_safe)
 
   [ "$pidproxy" == "" ] && echo $msg ABORT no pidproxy ... comes with supervisor : sv-get  && return 1
   [ "$pidpath" == "" ] && echo $msg ABORT no pidpath $pidpath && return 1
   [ "$mysqld_safe" == "" ] && echo $msg ABORT no mysql_safe && return 1

   cat << EOC
## http://supervisord.org/manual/current/subprocesses.html
##
##  try doing mysql config in more globally usable /etc/my.cnf rather than here ... 
##      --log-bin=logbin
##
[program:mysql]
command=$pidproxy $pidpath $mysqld_safe 
redirect_stderr=true
user=root
EOC
}

mysql-sv(){  sv-;sv-add $FUNCNAME- mysql.ini ; }





mysql-python-dir(){ echo $(local-base)/mysql-python/MySQL-python-$(mysql-python-ver) ; }
mysql-python-ver(){ echo 1.2.3c1 ; }
mysql-python-tgz(){ echo MySQL-python-$(mysql-python-ver).tar.gz ; }
mysql-python-url(){ echo http://downloads.sourceforge.net/project/mysql-python/mysql-python-test/$(mysql-python-ver)/$(mysql-python-tgz) ; }
mysql-python-cd(){  cd $(mysql-python-dir) ; }
mysql-python-get(){  

   local dir=$(dirname  $(mysql-python-dir))
   local nam=$(basename $(mysql-python-dir))
   mkdir -p $dir && cd $dir   
 
   local tgz=$(mysql-python-tgz)
   [ ! -f "$tgz" ] && curl -L -O $(mysql-python-url) 
   [ ! -d "$nam" ] && tar zxvf $tgz
}





