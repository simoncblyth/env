mysql-src(){    echo mysql/mysql.bash ; }
mysql-source(){ echo ${BASH_SOURCE:-$(env-home)/$(mysql-src)} ; }
mysql-vi(){     vim $(mysql-source) ; }
mysql-usage(){
  cat << EOU

   == VERSIONS ==

    Determine from   "echo status | mysql"     

        dybdb1   :   5.0.45-community-log MySQL Community Edition (GPL)
        belle7   :   5.0.77-log Source distribution
        cms01    :   4.1.22-log

    CAUTION :
        * user creation changed significantly between 4.1 and 5.0 


   === for mysql debugging (eg on trying to switch on logging ) ===

    Flavors of control ...
       * sv via pidproxy
       * mysql-start 
       * /sbin/service mysql start 

    CONSOLIDATION NEED ...  

   ===
   
    Traditional redhat control :

       sudo /sbin/service mysqld start
       sudo /sbin/service mysqld status
       sudo /sbin/service mysqld stop

    When network access is enabled the mysql port is $(local-port mysql)
      [blyth@cms01 rootmq]$ sudo lsof -i :$(local-port mysql)

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

    mysql-dumpall [basedir]

          CAUTION AS EACH db is dumped separately it is possible that the tables in 
          different DB will be inconsistent if one were to operate 
          in a writing related tables to separate DB manner ?
  


    mysql-stop
      
       sudo /opt/local/share/mysql5/mysql/mysql.server stop  
       ERROR! MySQL manager or server PID file could not be found!

       failing for want of pidof on OSX ?       
          sudo port install proctools 
               yielded pgrep/pfind/pkill but no pidof
       so cheat : 
          sudo ln -s /opt/local/bin/pgrep /opt/local/bin/pidof

       no joy, the problem is that the server pid file name is based of
       the hostname which tends to change ... 

          cd /opt/local/var/db/mysql5
          sudo cp  g4pb.local.pid $(hostname).pid


     Opening mysql to access from a remote webserver...

         1) set "bind-address" in mysql config of server 
   
         IPTABLES_PORT=3306 iptables-webopen-ip 140.###.###.###



   == remote mysql dump ==

      Following powercut cms01 is indisposed again...  so do remote dump 
     (have to use --skip-opt as do not have permission to lock the tables )

[blyth@belle7 ~]$ mysqldump --skip-opt testdb SimPmtSpec SimPmtSpecVld > SimPmtSpec.sql
Warning: mysqldump: ignoring option '--databases' due to invalid value 'testdb'
Warning: mysqldump: ignoring option '--databases' due to invalid value 'testdb'



   == set up passwords ==

      http://dev.mysql.com/doc/refman/5.1/en/default-privileges.html

mysql> SELECT User, Host, Password FROM mysql.user;
+------+-------------------+----------+
| User | Host              | Password |
+------+-------------------+----------+
| root | localhost         |          | 
| root | belle7.nuu.edu.tw |          | 
| root | 127.0.0.1         |          | 
+------+-------------------+----------+
3 rows in set (0.01 sec)

       SET PASSWORD FOR 'root'@'localhost' = PASSWORD('***');
       SET PASSWORD FOR 'root'@'belle7.nuu.edu.tw' = PASSWORD('***');
       SET PASSWORD FOR 'root'@'127.0.0.1' = PASSWORD('***');




    == Switching on logging ... ==

       * find that the logfile has to exist already 


100707 19:03:36  mysqld started
/usr/libexec/mysqld: File '/var/log/mysqld_out.log' not found (Errcode: 13)
100707 19:03:36 [ERROR] Could not use /var/log/mysqld_out.log for logging (error 13). Turning logging off for the whole duration of the MySQL server process. To turn it on again: fix the cause, shutdown the MySQL server and restart it.
100707 19:03:36  InnoDB: Started; log sequence number 0 44756
/usr/libexec/mysqld: ready for connections.
Version: '4.1.22-log'  socket: '/var/lib/mysql/mysql.sock'  port: 3306  Source distribution
[blyth@cms01 log]$ 
[blyth@cms01 log]$ sudo touch  /var/log/mysqld_out.log
[blyth@cms01 log]$ sudo chown mysql.mysql  /var/log/mysqld_out.log


   == Opening access to all with the password .. ==


     1) the firewall
           IPTABLES_PORT=3306 iptables-webopen

     2) db-grant DAYABAY 
           the version of mysql on cms01 4.1 does not have "create user"


   == how about opening resrtricted access to the log ... ==

     1) BUT tis very big ... as it swallows the mysqldump 
     
          tailing a logfile thru a web server ... more effort than gain 
              http://www.xinotes.org/notes/note/155/

     2) truncate/rotate/purge  the logs ... 

        not convenient to do with the db-backup-recover cronline, as that runs as me 
        and the logs are owned by mysql ... set up log rotation in separate root cronline

        using logrotate to manage the logs 


  == what happended to /etc/logrotate.d/mysql ==

    *  rpm -ql mysql-server  | grep log

     mysql appears not to be a good citizen :
         [blyth@belle7 logrotate.d]$ yum whatprovides "*/logrotate.d/*"  | grep mysql 


    * http://www.mail-archive.com/rhelv5-list@redhat.com/msg00781.html


/var/log/mysql/mysql.err /var/log/mysql/mysql.log /var/log/mysql/mysqld.err {
   monthly
   create 660 mysql mysql
   notifempty
   size 5M
   sharedscripts
   missingok
   postrotate
     /bin/kill -HUP `cat /var/run/mysqld/mysqld.pid`
   endscript
}


EOU
}





my-(){
  [ ~/.my.cnf -nt ~/.my/client.cnf ] && python $(env-home)/mysql/splitcnf.py 
  [ -z "$1" ] && ls -l ~/.my/   
  mysql --defaults-file=~/.my/${1:-client}.cnf
} 



mysql-logrotate-(){ cat << EOC
$(mysql-logpath) {
   daily 
   rotate 3

}
EOC
}

mysql-current-user(){
    echo status | mysql | perl -n -e 'm,Current user:\s*(\S*)$, && print $1 ' -
}

mysql-logpath(){ cfp- ; CFP_PATH=$(mysql-syscnf) cfp-getset mysqld log ; }
mysql-elogpath(){ cfp- ; CFP_PATH=$(mysql-syscnf) cfp-getset mysqld_safe err-log ; }
mysql-logsetup-(){  cat << EOS
# enter something like the below into $(mysql-syscnf) ... you can use mysql-sysedit
[mysqld]
log=$(mysql-logdir)/mysqld_out.log
EOS
}

mysql-ls(){
   cd $(mysql-logdir)
   ls -l *
}

mysql-logsetup(){
   local msg="=== $FUNCNAME :"
   local path=$(mysql-logpath)
   local logd=$(mysql-logdir)

   [ "$path" == "" ] && echo $msg ERROR you did not configure mysql for logging ... && $FUNCNAME- && return 1
   case $path in 
      $logd/*) echo $msg OK mysql-lopath $path is within $logd ;;
            *) echo $msg ERROR unexpected mysql-logpath $path ...not in $logd && return 1 ;; 
   esac
  [ ! -d "$logd" ] && sudo mkdir $logd && sudo chown -R mysql:mysql $logd
  [ ! -f "$path" ] && sudo touch $path && sudo chown mysql:mysql $path 
   
  mysql-ls
}

mysql-tail(){    sudo tail -f $(mysql-logpath) ; }
mysql-etail(){    sudo tail -f $(mysql-elogpath) ; }



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

mysql-logdir(){
  pkgr-
  case $(pkgr-cmd) in
    port) echo /opt/local/var/log/mysql5 ;;
       *) echo /var/log/mysql ;;
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

   local msg="=== $FUNCNAME :"
   echo $msg CAUTION this connects with private-val coordinates ... which may differ from the standard client section params in ~/.my.cnf used by the \"mysql\" client
   private-
   mysql-sh- $(private-val DATABASE_NAME) ; 
}

mysql-showdatabases(){
   echo show databases | mysql-sh- --skip-column-names
}

mysql-users(){
  echo select Host, User, Password, Select_priv from mysql.user \; | db-recover
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
   local iwd=$PWD
   local dir=$(mysql-bkpdir $*)
   mkdir -p $dir && cd $dir
   mysql-showdatabases | while read db_ ; do
       mysql-dump $db_ > $db_.sql
   done
   cd $iwd
}

mysql-syscnf(){
  pkgr-
  case $(pkgr-cmd) in
    yum) echo /etc/my.cnf ;;
      *) echo /etc/my.cnf
  esac
}
mysql-cnf(){ echo $HOME/.my.cnf ; }
mysql-edit(){    vi $(mysql-cnf) ; }
mysql-sysedit(){ sudo vi $(mysql-syscnf) ; }
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
command=$pidproxy $pidpath $mysqld_safe --defaults-file=$(mysql-syscnf)
redirect_stderr=true
priority=333
user=root
EOC
}

mysql-sv(){  sv-; $FUNCNAME- | sv-plus mysql.ini ; }





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





