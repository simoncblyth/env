# === func-gen- : sqlite/sqlite3 fgp sqlite/sqlite3.bash fgn sqlite3 fgh sqlite
sqlite3-src(){      echo sqlite/sqlite3.bash ; }
sqlite3-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sqlite3-src)} ; }
sqlite3-vi(){       vi $(sqlite3-source) ; }
sqlite3-env(){      elocal- ; }
sqlite3-usage(){ cat << EOU

SQLITE3
========

* http://www.sqlite.org/download.html

yum depends on sqlite3



regexp extension
----------------

* https://github.com/ralight/sqlite3-pcre


compound statements, group by statements
------------------------------------------

::

    select distinct tag as t from tags order by tag ;
    select distinct tag as t from tags where t not in ( select distinct name from wiki ) order by tag ;
    select name, count(tag) as n from tags group by tag order by n desc ;



timezone documentation extracts
--------------------------------

* https://www.sqlite.org/lang_datefunc.html

unixepoch
~~~~~~~~~~

The "unixepoch" modifier (11) only works if it immediately follows a timestring
in the DDDDDDDDDD format. This modifier causes the DDDDDDDDDD to be interpreted
not as a Julian day number as it normally would be, but as Unix Time - the
number of seconds since 1970. If the "unixepoch" modifier does not follow a
timestring of the form DDDDDDDDDD which expresses the number of seconds since
1970 or if other modifiers separate the "unixepoch" modifier from prior
DDDDDDDDDD then the behavior is undefined. 

localtime
~~~~~~~~~~~

The "localtime" modifier (12) assumes the time string to its left is in
Universal Coordinated Time (UTC) and adjusts the time string so that it
displays localtime. If "localtime" follows a time that is not UTC, then the
behavior is undefined. 

utc
~~~~

The "utc" is the opposite of "localtime". "utc" assumes
that the string to its left is in the local timezone and adjusts that string to
be in UTC. If the prior string is not in localtime, then the result of "utc" is
undefined.


datetime timezone handling 
---------------------------

Documentation on sqlite3 timezone handling seems incorrect to me.

* :google:`sqlite3 lang_datefunc utc and localtime modifiers documentation wrong`


Fields std and loc holds 2 epoch second "ints" one being utc the other 
8hrs ahead::

    sqlite> select loc, std, (loc-std)/60/60 from log ;
    loc         std         (loc-std)/60/60
    ----------  ----------  ---------------
    1417757957  1417729157  8      


Corresponding strings in locdt and stddt::

    sqlite> select locdt, loc, stddt, std from log ;
    locdt                loc         stddt                std       
    -------------------  ----------  -------------------  ----------
    2014-12-05 13:39:17  1417757957  2014-12-05 05:39:17  1417729157


To express std into a local datetime string, need two localtime modifiers::

    sqlite> select std, datetime(std,'unixepoch'), datetime(std,'unixepoch','localtime'), datetime(std,'unixepoch','localtime','localtime')  from log ;
    std         datetime(std,'unixepoch')  datetime(std,'unixepoch','localtime')  datetime(std,'unixepoch','localtime','localtime')
    ----------  -------------------------  -------------------------------------  -------------------------------------------------
    1417729157  2014-12-04 21:39:17        2014-12-05 05:39:17                    2014-12-05 13:39:17     



Seems that that unixepoch modifier does a timezone shunt, that have to compensare for::

    sqlite> select loc, datetime(loc,'unixepoch'), datetime(loc,'unixepoch','localtime'), locdt  from log ;
    loc         datetime(loc,'unixepoch')  datetime(loc,'unixepoch','localtime')  locdt              
    ----------  -------------------------  -------------------------------------  -------------------
    1417757957  2014-12-05 05:39:17        2014-12-05 13:39:17                    2014-12-05 13:39:17


Unix time
~~~~~~~~~~~~

* http://en.wikipedia.org/wiki/Unix_time

Unix time (a.k.a. POSIX time or Epoch time) is a system for describing instants
in time, defined as the number of seconds that have elapsed since 00:00:00
Coordinated Universal Time (UTC), Thursday, 1 January 1970 not
counting leap seconds


Example : Trac wiki modification times
----------------------------------------

::

    sqlite> select version, datetime(time, 'unixepoch') from wiki where name='3D' ;
    version     datetime(time, 'unixepoch')
    ----------  ---------------------------
    1           2009-02-18 03:20:11        
    2           2009-02-18 03:32:53        
    3           2009-02-18 05:33:07        
    4           2009-03-17 10:23:16        

    sqlite> select version, datetime(time, 'unixepoch', 'localtime') from wiki where name='3D' ;
    version     datetime(time, 'unixepoch', 'localtime')
    ----------  ----------------------------------------
    1           2009-02-18 11:20:11                     
    2           2009-02-18 11:32:53                     
    3           2009-02-18 13:33:07                     
    4           2009-03-17 18:23:16                     
    sqlite> 





DATETIME fields does it make a difference
-------------------------------------------

::

    sqlite> create table t(dt DATETIME default current_timestamp, id integer);
    sqlite> insert into t values(null,1) ;
    sqlite> select * from t ;
    dt          id        
    ----------  ----------
                1         
    sqlite> select current_timestamp ;
    current_timestamp  
    -------------------
    2014-12-05 06:11:11
    sqlite> insert into t(id) values (10) ;
    sqlite> select * from t ;
    dt          id        
    ----------  ----------
                1         
    2014-12-05  10        
    sqlite> .width 20 10
    sqlite> select * from t ;
    dt                    id        
    --------------------  ----------
                          1         
    2014-12-05 06:12:56   10        

    sqlite> select datetime(dt,'localtime') from t where id=10 ;
    2014-12-05 14:12:56 

    sqlite> pragma table_info(t);
    cid                   name        type        notnull     dflt_value         pk        
    --------------------  ----------  ----------  ----------  -----------------  ----------
    0                     dt          DATETIME    0           current_timestamp  0         
    1                     id          integer     0                              0       




EOU
}
sqlite3-dir(){ echo $(local-base)/env/sqlite/$(sqlite3-name) ; }
sqlite3-prefix(){ echo $(local-base)/env ; }

sqlite3-cd(){  cd $(sqlite3-dir); }
sqlite3-mate(){ mate $(sqlite3-dir) ; }
sqlite3-name(){ echo sqlite-autoconf-3080002 ; }
sqlite3-url(){  echo http://www.sqlite.org/2013/$(sqlite3-name).tar.gz ; }
sqlite3-get(){
   local dir=$(dirname $(sqlite3-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(sqlite3-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && tar zxvf $tgz

}

sqlite3-path(){ PATH=$(sqlite3-prefix)/bin:$PATH ; }
sqlite3--(){ $(sqlite3-prefix)/bin/sqlite3 $* ; }

sqlite3-build(){
  sqlite3-configure
  sqlite3-make
  sqlite3-install
}


sqlite3-configure(){
   sqlite3-cd
   ./configure --prefix=$(sqlite3-prefix) 
}

sqlite3-make(){
   sqlite3-cd
   make 
}

sqlite3-install(){
   sqlite3-cd
   sudo make install
}





