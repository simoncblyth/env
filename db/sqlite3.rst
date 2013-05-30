SQLite3 
=========

Looking for elegant approaches for hooking up related tables

* http://stackoverflow.com/questions/13244393/sqlite-insert-or-ignore-and-return-original-rowid


insert or replace
-----------------

::

    sqlite> CREATE TABLE colors (label text primary key);
    sqlite> CREATE UNIQUE INDEX idx_something ON colors (label) ;
    sqlite> insert or replace into colors values ('red') ;
    sqlite> insert or replace into colors values ('green') ;
    sqlite> insert or replace into colors values ('blue') ;
    sqlite> select rowid, * from colors ;
    rowid       label     
    ----------  ----------
    1           red       
    2           green     
    3           blue      
    sqlite> insert or replace into colors values ('red') ;
    sqlite> select rowid, * from colors ;
    rowid       label     
    ----------  ----------
    2           green     
    3           blue      
    4           red       
    sqlite> 



`insert or replace` Keeps incrementing
---------------------------------------

::

    sqlite> create table B (id integer primary key autoincrement, label text, constraint label_unique unique (label) );
    sqlite> insert or replace into B values (NULL,"red");
    sqlite> insert or replace into B values (NULL,"red");
    sqlite> select * from B ;
    id          label     
    ----------  ----------
    2           red       

    sqlite> create table C (id integer primary key, label text, constraint label_unique unique (label) );
    sqlite> insert or replace into C values (NULL,"red");
    sqlite> insert or replace into C values (NULL,"red");
    sqlite> select * from C ;
    id          label     
    ----------  ----------
    2           red       

    sqlite> create table D (label text primary key) ;
    sqlite> insert or replace into D values ("red");
    sqlite> insert or replace into D values ("red");
    sqlite> select rowid,* from D ;
    rowid       label     
    ----------  ----------
    2           red       


`On conflict replace`
----------------------

* http://www.sqlite.org/lang_conflict.html

::

    sqlite> create table E (label text primary key on conflict replace) ;
    sqlite> insert into E values ("red") ;
    sqlite> insert into E values ("red") ;
    sqlite> select rowid, * from E ;
    rowid       label     
    ----------  ----------
    2           red       



`On conflict ignore` on the column definition
-----------------------------------------------

Succeeds to not increment when pre-existing, but `last_insert_rowid` 
doesnt provide the key to the row that you didnt just add::

    sqlite> create table F (label text primary key on conflict ignore) ;
    sqlite> insert into F values ("red") ;
    sqlite> insert into F values ("red") ;
    sqlite> insert into F values ("green") ;
    sqlite> insert into F values ("blue") ;
    sqlite> insert into F values ("red") ;
    sqlite> select rowid, * from F ;
    rowid       label     
    ----------  ----------
    1           red       
    2           green     
    3           blue      
    sqlite> 
    sqlite> select last_insert_rowid() ;
    last_insert_rowid()
    -------------------
    3                  

* http://www.sqlite.org/c3ref/last_insert_rowid.html

`insert or ignore` on the insert
-----------------------------------

Same effect with `insert or ignore`

::

    sqlite> create table G (label text primary key) ;
    sqlite> insert or ignore into G values ("red") ;
    sqlite> insert or ignore into G values ("red") ;
    sqlite> insert or ignore into G values ("green") ;
    sqlite> insert or ignore into G values ("green") ;
    sqlite> insert or ignore into G values ("green") ;
    sqlite> insert or ignore into G values ("green") ;
    sqlite> insert or ignore into G values ("green") ;
    sqlite> insert or ignore into G values ("blue") ;
    sqlite> insert or ignore into G values ("red") ;
    sqlite> select rowid, * from G ;
    rowid       label     
    ----------  ----------
    1           red       
    2           green     
    3           blue      
    sqlite> select last_insert_rowid() ;
    last_insert_rowid()
    -------------------
    3                  





