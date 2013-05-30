# === func-gen- : mysql/mysqlrpm fgp mysql/mysqlrpm.bash fgn mysqlrpm fgh mysql
mysqlrpm-src(){      echo mysql/mysqlrpm.bash ; }
mysqlrpm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysqlrpm-src)} ; }
mysqlrpm-vi(){       vi $(mysqlrpm-source) ; }
mysqlrpm-env(){      elocal- ; }
mysqlrpm-usage(){ cat << EOU

MySQL from rpms
===================

* http://dev.mysql.com/doc/refman/5.0/en/source-installation.html
* http://dev.mysql.com/doc/refman/5.0/en/linux-installation-rpm.html
* http://downloads.mysql.com/archives.php?p=mysql-5.0&v=5.0.45

belle1 installation
----------------------

server install
~~~~~~~~~~~~~~

Using the version *5.0.45* to precisely match that used on the official servers::

	[blyth@belle1 env]$ sudo rpm -i MySQL-server-community-5.0.45-0.rhel5.i386.rpm
	[sudo] password for blyth: 
	error: Failed dependencies:
		perl(DBI) is needed by MySQL-server-community-5.0.45-0.rhel5.i386

	[blyth@belle1 env]$ sudo yum install perl-DBI

	[blyth@belle1 env]$  sudo rpm -i MySQL-server-community-5.0.45-0.rhel5.i386.rpm
	PLEASE REMEMBER TO SET A PASSWORD FOR THE MySQL root USER !
	To do so, start the server, then issue the following commands:
	/usr/bin/mysqladmin -u root password 'new-password'
	/usr/bin/mysqladmin -u root -h belle1.nuu.edu.tw password 'new-password'
	See the manual for more instructions.
	Please report any problems with the /usr/bin/mysqlbug script!

	The latest information about MySQL is available on the web at
	http://www.mysql.com
	Support MySQL by buying support/licenses at http://shop.mysql.com
	Starting MySQL[  OK  ]
	Giving mysqld 2 seconds to start
	[blyth@belle1 env]$ 



client install
~~~~~~~~~~~~~~~

::

	[blyth@belle1 env]$ sudo rpm -i MySQL-client-community-5.0.45-0.rhel5.i386.rpm    # mysqladmin is client rpm so must install that first

password setup
~~~~~~~~~~~~~~~~

* http://dev.mysql.com/doc/refman/5.0/en/default-privileges.html


::

	[blyth@belle1 env]$ password=***
	[blyth@belle1 env]$ /usr/bin/mysqladmin -u root password '$password'
	[blyth@belle1 env]$ /usr/bin/mysqladmin -u root -h belle1.nuu.edu.tw password '$password'
                                  ## oops that accidentally set password to dollar-password as used single quotes

        [blyth@belle1 env]$  /usr/bin/mysqladmin -u root -p password "$password"         ## fix with "-p" 
        Enter password: 
	[blyth@belle1 env]$ /usr/bin/mysqladmin -u root -h belle1.nuu.edu.tw -p password "$password"
        Enter password: 




	[blyth@belle1 env]$ mysql -u root -h 127.0.0.1 -p        
	Enter password: 
	Welcome to the MySQL monitor.  Commands end with ; or \g.
	Your MySQL connection id is 24
	Server version: 5.0.45-community MySQL Community Edition (GPL)

	mysql> select Host,User,Password from mysql.user ;
	+-------------------+------+-------------------------------------------+
	| Host              | User | Password                                  |
	+-------------------+------+-------------------------------------------+
	| localhost         | root | *3AAC99F30A50D7D8D0***                    | 
	| belle1.nuu.edu.tw | root | *3AAC99F30A50D7D8D0***                    | 
	| 127.0.0.1         | root |                                           |    ## curiously loopback password still blank 
	+-------------------+------+-------------------------------------------+
	3 rows in set (0.00 sec)

	mysql> SET PASSWORD FOR 'root'@'127.0.0.1' = PASSWORD('***');
	Query OK, 0 rows affected (0.00 sec)

	mysql> select Host,User,Password from mysql.user ;
	+-------------------+------+-------------------------------------------+
	| Host              | User | Password                                  |
	+-------------------+------+-------------------------------------------+
	| localhost         | root | *3AAC99F30A50D7D8D***                     | 
	| belle1.nuu.edu.tw | root | *3AAC99F30A50D7D8D***                     | 
	| 127.0.0.1         | root | *3AAC99F30A50D7D8D***                     | 
	+-------------------+------+-------------------------------------------+
	3 rows in set (0.00 sec)


Create :file:`~/.my.cnf`
~~~~~~~~~~~~~~~~~~~~~~~~~~

Add client section, remember to protect it::

	[blyth@belle1 e]$ chmod go-rw ~/.my.cnf 



OOPS : mysqlhotcopy not working
---------------------------------

perlmodule DBD::MySQL needed for mysqlhotcopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Yuck CPAN

* http://search.cpan.org/dist/DBD-mysql/lib/DBD/mysql.pm#INSTALLATION

::

	[root@belle1 ~]# perl -mDBD::mysql -e ''
	Can't locate DBD/mysql.pm in @INC (@INC contains: /usr/lib/perl5/site_perl/5.8.8/i386-linux-thread-multi /usr/lib/perl5/site_perl/5.8.8 /usr/lib/perl5/site_perl /usr/lib/perl5/vendor_perl/5.8.8/i386-linux-thread-multi /usr/lib/perl5/vendor_perl/5.8.8 /usr/lib/perl5/vendor_perl /usr/lib/perl5/5.8.8/i386-linux-thread-multi /usr/lib/perl5/5.8.8 .).
	BEGIN failed--compilation aborted.  


yum fails, maybe as are using non-standard version for dybdb1 correspondence::

	[root@belle1 ~]# yum install perl-DBD-MySQL 
	Loaded plugins: kernel-module
	Setting up Install Process
	Resolving Dependencies
	There are unfinished transactions remaining. You might consider running yum-complete-transaction first to finish them.
	The program yum-complete-transaction is found in the yum-utils package.
	--> Running transaction check
	---> Package perl-DBD-MySQL.i386 0:3.0007-1.fc6 set to be updated
	--> Processing Dependency: libmysqlclient.so.15 for package: perl-DBD-MySQL
	--> Processing Dependency: libmysqlclient.so.15(libmysqlclient_15) for package: perl-DBD-MySQL
	--> Running transaction check
	---> Package mysql.i386 0:5.0.95-5.el5_9 set to be updated
	--> Processing Conflict: mysql conflicts MySQL
	--> Finished Dependency Resolution
	mysql-5.0.95-5.el5_9.i386 from sl-security has depsolving problems
	  --> mysql conflicts with MySQL-server-community
	Beginning Kernel Module Plugin
	Finished Kernel Module Plugin
	Error: mysql conflicts with MySQL-server-community
	 You could try using --skip-broken to work around the problem
	 You could try running: package-cleanup --problems
				package-cleanup --dupes
				rpm -Va --nofiles --nodigest
	The program package-cleanup is found in the yum-utils package.
	[root@belle1 ~]# 


install the shared too
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

	[blyth@belle1 env]$ rpm -i MySQL-shared-community-5.0.45-0.rhel5.i386.rpm
	error: can't create transaction lock on /var/lib/rpm/__db.000
	[blyth@belle1 env]$ sudo rpm -i MySQL-shared-community-5.0.45-0.rhel5.i386.rpm
	[sudo] password for blyth: 
	[blyth@belle1 env]$ 



try again after installing the shared succeeds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

	[root@belle1 ~]# yum install perl-DBD-MySQL 
	...
	Installed:
	  perl-DBD-MySQL.i386 0:3.0007-1.fc6                                                                                                                                                                                                                              
	Complete!
	[root@belle1 ~]# 
	[root@belle1 ~]# perl -mDBD::mysql -e ''
	[root@belle1 ~]# 


EOU
}
mysqlrpm-dir(){ echo $(local-base)/env/mysql ; }
mysqlrpm-cd(){  cd $(mysqlrpm-dir); }
mysqlrpm-mate(){ mate $(mysqlrpm-dir) ; }
mysqlrpm-get(){
   local dir=$(dirname $(mysqlrpm-dir)) &&  mkdir -p $dir && cd $dir

  
   $FUNCNAME-server
   $FUNCNAME-client
   $FUNCNAME-shared

}

mysqlrpm-version(){
   echo 5.0.45
}
mysqlrpm-url(){
   echo http://downloads.mysql.com/archives/mysql-5.0/MySQL-${1}-community-$(mysqlrpm-version)-0.rhel5.i386.rpm
}

mysqlrpm-get-server(){
   local url=$(mysqlrpm-url server)
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -O $url
}
mysqlrpm-get-client(){
   local url=$(mysqlrpm-url client)
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -O $url
}
mysqlrpm-get-shared(){
   local url=$(mysqlrpm-url shared)
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -O $url
}





mysqlrpm-rpm-check(){
   rpm -qpl $1 | while read path ; do  [ -f $path ] && echo would clobber $path  ; done 
}


mysqlrpm-check(){
  [ $(uname -m) == "i686" ] && echo 32 bit 
  [ $(uname -m) == "x86_64" ] && echo 64 bit 
}

