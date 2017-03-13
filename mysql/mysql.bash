mysql-src(){    echo mysql/mysql.bash ; }
mysql-source(){ echo ${BASH_SOURCE:-$(env-home)/$(mysql-src)} ; }
mysql-vi(){     vim $(mysql-source) ; }
mysql-usage(){ cat << EOU

MYSQL
=====

versions
----------

Determine from::   

   echo status | mysql

* dybdb1 : 5.0.45-community-log MySQL Community Edition (GPL)
* belle7 : 5.0.77-log Source distribution
* cms01 : 4.1.22-log

* NB user creation changed significantly between 4.1 and 5.0 


macports MySQL
------------------

* https://trac.macports.org/wiki/howto/MySQL

Need to setup PATH to find mysql client and perhaps change the client section 
of cnf to point at desired DB.

* /opt/local/lib/mysql56/bin/mysql
* adjust ~/.my.cnf


control options
-----------------

Flavors of control ...

* sv via pidproxy
* mysql-start 
* /sbin/service mysql start 
* launchctl on OSX ?
* redhat service

redhat service
~~~~~~~~~~~~~~~
   
::

       sudo /sbin/service mysqld start
       sudo /sbin/service mysqld status
       sudo /sbin/service mysqld stop

port used
------------

When network access is enabled the mysql port is *local-port mysql*::

    [blyth@belle7 ~]$ sudo lsof -i :3306
    [sudo] password for blyth: 
    COMMAND   PID  USER   FD   TYPE    DEVICE SIZE NODE NAME
    mysqld  24923 mysql   11u  IPv4 157220234       TCP *:mysql (LISTEN)
    mysqld  24923 mysql   14u  IPv4 270422956       TCP localhost.localdomain:mysql->localhost.localdomain:37026 (ESTABLISHED)
    mysqld  24923 mysql   17u  IPv4 270422958       TCP localhost.localdomain:mysql->localhost.localdomain:37027 (ESTABLISHED)
    mysqld  24923 mysql   19u  IPv4 270434694       TCP localhost.localdomain:mysql->localhost.localdomain:36868 (ESTABLISHED)
    python  28294 blyth    4u  IPv4 270422951       TCP belle7.nuu.edu.tw:47411->dybdb2.ihep.ac.cn:mysql (ESTABLISHED)
    python  28294 blyth    5u  IPv4 270422953       TCP belle7.nuu.edu.tw:47412->dybdb2.ihep.ac.cn:mysql (ESTABLISHED)
    python  28294 blyth    6u  IPv4 270422955       TCP localhost.localdomain:37026->localhost.localdomain:mysql (ESTABLISHED)
    python  28294 blyth    7u  IPv4 270422957       TCP localhost.localdomain:37027->localhost.localdomain:mysql (ESTABLISHED)
    python  28294 blyth    8u  IPv4 270434693       TCP localhost.localdomain:36868->localhost.localdomain:mysql (ESTABLISHED)


FUNCTIONS
------------


*mysql-sv*
        Add to supervisor control, needs to be configured to run as root


*mysql-triplet-edit "mysqld|log-bin|binlog"*


*mysql-showdatabase*

*mysql-dumpall basedir*
       CAUTION AS EACH db is dumped separately it is possible that the tables in 
       different DB will be inconsistent if one were to operate 
       in a writing related tables to separate DB manner ?

*mysql-stop*
       sudo /opt/local/share/mysql5/mysql/mysql.server stop  
       ERROR! MySQL manager or server PID file could not be found!

       failing for want of pidof on OSX ?::

           sudo port install proctools 
               yielded pgrep/pfind/pkill but no pidof

       so cheat : 

           sudo ln -s /opt/local/bin/pgrep /opt/local/bin/pidof

       no joy, the problem is that the server pid file name is based of
       the hostname which tends to change::

          cd /opt/local/var/db/mysql5
          sudo cp  g4pb.local.pid *hostname*.pid


MYSQL ACCESS
--------------


Opening mysql to access from a remote webserver...

#. set "bind-address" in mysql config of server and open the port::
   
         IPTABLES_PORT=3306 iptables-webopen-ip 140.###.###.###


remote mysql dump
-------------------

Following powercut cms01 is indisposed again, so do remote dump 
(have to use *--skip-opt* as do not have permission to lock the tables )

::

    [blyth@belle7 ~]$ mysqldump --skip-opt testdb SimPmtSpec SimPmtSpecVld > SimPmtSpec.sql
    Warning: mysqldump: ignoring option '--databases' due to invalid value 'testdb'
    Warning: mysqldump: ignoring option '--databases' due to invalid value 'testdb'



set up passwords
------------------

* http://dev.mysql.com/doc/refman/5.1/en/default-privileges.html

::

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


useful selects to debug permissions 
------------------------------------

::

    mysql> select Host, User, Password, Select_priv, Insert_priv,Update_priv,Drop_priv, File_priv from mysql.user ;
    +-----------------------+---------+-------------------------------------------+-------------+-------------+-------------+-----------+-----------+
    | Host                  | User    | Password                                  | Select_priv | Insert_priv | Update_priv | Drop_priv | File_priv |
    +-----------------------+---------+-------------------------------------------+-------------+-------------+-------------+-----------+-----------+




Switching on logging
----------------------

* find that the logfile has to exist already 


::

    100707 19:03:36  mysqld started
    /usr/libexec/mysqld: File '/var/log/mysqld_out.log' not found (Errcode: 13)
    100707 19:03:36 [ERROR] Could not use /var/log/mysqld_out.log for logging (error 13). Turning logging off for the whole duration of the MySQL server process. To turn it on again: fix the cause, shutdown the MySQL server and restart it.
    100707 19:03:36  InnoDB: Started; log sequence number 0 44756
    /usr/libexec/mysqld: ready for connections.
    Version: '4.1.22-log'  socket: '/var/lib/mysql/mysql.sock'  port: 3306  Source distribution
    [blyth@cms01 log]$ 
    [blyth@cms01 log]$ sudo touch  /var/log/mysqld_out.log
    [blyth@cms01 log]$ sudo chown mysql.mysql  /var/log/mysqld_out.log


how about opening resrtricted access to the log 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. BUT tis very big, as it swallows the mysqldump 

#. truncate/rotate/purge the logs 

   not convenient to do with the db-backup-recover cronline, as that runs as me 
   and the logs are owned by mysql ... set up log rotation in separate root cronline
   using logrotate to manage the logs 


what happended to /etc/logrotate.d/mysql
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    rpm -ql mysql-server  | grep log

mysql appears not to be a good citizen::

         [blyth@belle7 logrotate.d]$ yum whatprovides "*/logrotate.d/*"  | grep mysql 


Opening access to all with the password 
-----------------------------------------

#. the firewall::

    IPTABLES_PORT=3306 iptables-webopen

#. *db-grant DAYABAY* 
   
   * the version of mysql on cms01 4.1 does not have "create user"


note regards pausing mysql slave replication 
---------------------------------------------

* http://dev.mysql.com/doc/refman/5.0/en/replication-howto-masterbaseconfig.html

     * set master server-id to 0 to disable slave connections 
    


OSX macports mysql5-server
-----------------------------

::

    simon:~ blyth$ sudo port contents mysql5-server
    Password:
    Warning: port definitions are more than two weeks old, consider using selfupdate
    Port mysql5-server contains:
      /Library/LaunchDaemons/org.macports.mysql5.plist
      /opt/local/etc/LaunchDaemons/org.macports.mysql5/mysql5.wrapper
      /opt/local/etc/LaunchDaemons/org.macports.mysql5/org.macports.mysql5.plist
      /opt/local/var/db/mysql5/.turd_mysql5-server
      /opt/local/var/log/mysql5/.turd_mysql5-server
      /opt/local/var/run/mysql5/.turd_mysql5-server
    simon:~ blyth$ 

Curious loading via macports::

    simon:~ blyth$ sudo port load mysql5-server

Note a hostname specific pid file, that will cause issues::

    simon:~ blyth$ ps aux | grep mysql5
    _mysql    2442   0.0  0.7   114492  14068   ??  S     6:31pm   0:00.20 /opt/local/libexec/mysqld --basedir=/opt/local --datadir=/opt/local/var/db/mysql5 --user=_mysql --log-error=/opt/local/var/db/mysql5/simon.phys.ntu.edu.tw.err --pid-file=/opt/local/var/db/mysql5/simon.phys.ntu.edu.tw.pid
    root      2389   0.0  0.0    75944    784   ??  S     6:31pm   0:00.06 /bin/sh /opt/local/lib/mysql5/bin/mysqld_safe --datadir=/opt/local/var/db/mysql5 --pid-file=/opt/local/var/db/mysql5/simon.phys.ntu.edu.tw.pid
    root      2380   0.0  0.0    75428    760   ??  Ss    6:31pm   0:00.02 /opt/local/bin/daemondo --label=mysql5 --start-cmd /opt/local/etc/LaunchDaemons/org.macports.mysql5/mysql5.wrapper start ; --stop-cmd /opt/local/etc/LaunchDaemons/org.macports.mysql5/mysql5.wrapper stop ; --restart-cmd /opt/local/etc/LaunchDaemons/org.macports.mysql5/mysql5.wrapper restart ; --pid=none
    simon:~ blyth$ 


OSX Mavericks/Yosemite MySQL 
------------------------------

* https://trac.macports.org/wiki/howto/MAMP


OSX Macports MySQL56 
----------------------

* https://trac.macports.org/wiki/howto/MySQL

::

    simon:~ blyth$ sudo port install mysql56-server
    Password:
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Computing dependencies for mysql56-server
    --->  Dependencies to be installed: mysql56 mysql_select tcp_wrappers
    --->  Fetching archive for mysql_select
    --->  Attempting to fetch mysql_select-0.1.2_0.darwin_13.noarch.tbz2 from http://packages.macports.org/mysql_select
    --->  Attempting to fetch mysql_select-0.1.2_0.darwin_13.noarch.tbz2.rmd160 from http://packages.macports.org/mysql_select
    --->  Installing mysql_select @0.1.2_0
    --->  Activating mysql_select @0.1.2_0
    --->  Cleaning mysql_select
    --->  Fetching archive for tcp_wrappers
    --->  Attempting to fetch tcp_wrappers-20_2.darwin_13.x86_64.tbz2 from http://packages.macports.org/tcp_wrappers
    --->  Attempting to fetch tcp_wrappers-20_2.darwin_13.x86_64.tbz2.rmd160 from http://packages.macports.org/tcp_wrappers
    --->  Installing tcp_wrappers @20_2
    --->  Activating tcp_wrappers @20_2
    --->  Cleaning tcp_wrappers
    --->  Fetching archive for mysql56
    --->  Attempting to fetch mysql56-5.6.23_0.darwin_13.x86_64.tbz2 from http://packages.macports.org/mysql56
    --->  Attempting to fetch mysql56-5.6.23_0.darwin_13.x86_64.tbz2 from http://jog.id.packages.macports.org/macports/packages/mysql56
    --->  Attempting to fetch mysql56-5.6.23_0.darwin_13.x86_64.tbz2 from http://lil.fr.packages.macports.org/mysql56
    --->  Fetching distfiles for mysql56
    --->  Attempting to fetch mysql-5.6.23.tar.gz from http://mysql.ntu.edu.tw/Downloads/MySQL-5.6
    --->  Attempting to fetch mysql-5.6.23.tar.gz from http://mysql.cdpa.nsysu.edu.tw/Downloads/MySQL-5.6
    --->  Attempting to fetch mysql-5.6.23.tar.gz from http://ftp.iij.ad.jp/pub/db/mysql/Downloads/MySQL-5.6
    --->  Attempting to fetch mysql-5.6.23.tar.gz from http://ftp.jaist.ac.jp/pub/mysql/Downloads/MySQL-5.6
    --->  Attempting to fetch mysql-5.6.23.tar.gz from http://cjj.kr.distfiles.macports.org/mysql56
    --->  Verifying checksums for mysql56                                            
    --->  Extracting mysql56
    --->  Applying patches to mysql56
    --->  Configuring mysql56
    Error: org.macports.configure for port mysql56 returned: configure failure: command execution failed
    Error: Failed to install mysql56
    Please see the log file for port mysql56 for details:
        /opt/local/var/macports/logs/_opt_local_var_macports_sources_rsync.macports.org_release_tarballs_ports_databases_mysql56/mysql56/main.log
    Error: The following dependencies were not installed: mysql56
    To report a bug, follow the instructions in the guide:
        http://guide.macports.org/#project.tickets
    Error: Processing of port mysql56-server failed

::

    simon:~ blyth$ sudo port install mysql56-server
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Computing dependencies for mysql56-server
    --->  Dependencies to be installed: mysql56
    --->  Configuring mysql56
    Error: org.macports.configure for port mysql56 returned: configure failure: command execution failed
    Error: Failed to install mysql56
    Please see the log file for port mysql56 for details:
        /opt/local/var/macports/logs/_opt_local_var_macports_sources_rsync.macports.org_release_tarballs_ports_databases_mysql56/mysql56/main.log
    Error: The following dependencies were not installed: mysql56
    To report a bug, follow the instructions in the guide:
        http://guide.macports.org/#project.tickets
    Error: Processing of port mysql56-server failed
    simon:~ blyth$ 


* installing mysql56 and then  mysql56-server succeeds, appararently some problem with dependency auto-installing


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
  cat << EON

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

EON

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






