mysql-src(){    echo mysql/mysql.bash ; }
mysql-source(){ echo ${BASH_SOURCE:-$(env-home)/$(mysql-src)} ; }
mysql-vi(){     vim $(mysql-source) ; }
mysql-usage(){
  cat << EOU


    See also 
           dj-vi


EOU
}

mysql-env(){
   pkgr-
  case $(pkgr-cmd) in
    port) PATH=/opt/local/lib/mysql5/bin:$PATH ;;
  esac
}


mysql-five(){
   [ "$(uname)" == "Darwin" ] && echo 5 ; 
}

mysql-sh-(){
   private-
   mysql$(mysql-five) --user $(private-val DATABASE_USER) --password=$(private-val DATABASE_PASSWORD) $1
}

mysql-sh(){
   private-
   mysql-sh- $(private-val DATABASE_NAME) ; 
}


mysql-ini(){
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



