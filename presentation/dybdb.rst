DYB DB
=========

Remit
-------

* setup
* design
* development


Spelunking History 
-------------------- 

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database
* http://dayabay.ihep.ac.cn/tracs/dybsvn/tags?q=%27DB%27
* http://dayabay.ihep.ac.cn/tracs/dybsvn/tags?q=%27Database%27

* http://dayabay.ihep.ac.cn/tracs/dybsvn/milestone/DBI_Integration
* http://dayabay.ihep.ac.cn/DocDB/0062/006244/002/OfflineDBTutorial.pdf


* http://dayabay.ihep.ac.cn/DocDB/0056/005642/001/IntroducingDybDBI.pdf
* http://dayabay.ihep.ac.cn/DocDB/0052/005274/001/DBSetup.pdf

Eliminate “source” duplication , standardizes DBI data objects / tables


Best Sources
-----------------

* http://dayabay.bnl.gov/oum/sop/dbops/#objectives 

Whats included
-------------------

* venerable C++ DBI, inherited from MINOS : based on ROOT TSQL, using MySQL  
* DBCONF envvar config approach, avoids juggling 
* DybDbi : python interface to control C++ DBI 

  * users will always use the python interface 
  * almost no one will use a C++ Database service 

* generation of MySQL table definitions from spec files (CSV based format
  specifying fields and types).  


* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/DatabaseSvc

  * C++ DatabaseSvc the no one ever used (?)

* python script general tool : enabled non-expert users to use mysql 

  * http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/DybPython/python/DybPython/db.py

* documentation, SOP: Standard Operating Procedures 

  * http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Documentation/OfflineUserManual/tex/database/database_interface.tex
  * http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Documentation/OfflineUserManual/tex/database/database_interface.tex

* backup system 
* scrapers



Validity Range Policy 
~~~~~~~~~~~~~~~~~~~~~~~


Interface needs to be smart
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/524

* eg jobs should not connect to DB for every event, if their time stamps are within a range fo which the current
  held values are valid 

We use DBI because it is supposed to be 'smart' and cache the result of
database calls. It should not call the database for every event. If it does
call the database every event, then it must be fixed. 


Feature Ticket 607 : define and document Standard Operating Procedures for DB updating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/607

Non Direct experience
~~~~~~~~~~~~~~~~~~~~~~~

* replication chains 

spec file auto conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* really auto : users just need to commit to SVN 

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/DybDbi/cmt/requirements

Testing updates
~~~~~~~~~~~~~~~~~~

* https://dayabay.bnl.gov/oum/sop/dbtest/#sop-dbtest

DybDbi
~~~~~~~~

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/DybDbi

Insertdate fastforwarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/844


Considerations
----------------

* monitoring tables (HV, temperatures) scraped from DCS, 
  automatically copies and write into DB 

  * very different lifecycle to calibrations : should be in different DB 

* handcrafting DB tables and "Row" class C++ code is an recipe for bugs
  and a maintenance nightmare : must generate table SQL and C++ code from 
  a common spec file 

  * http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/610
  * http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/DybDbi/spec
 
* MySQL INNODB vs MYISAM 
* backup 
* corruption recovery plan : experience
* users do stangest things : eg table ~200 times larger than it needs to be  
* bad calibrations happen : workflow needs to cope 
* automated testing of calibrations
* SOP documentatiom : https://dayabay.bnl.gov/oum/sop/dbops/#objectives
* rollback 
* auditable 
* reproducibility of every run ? 

  * demands being able to return to any previous state of the DB 
    (means never delete, only add), or multiple DB 
  * also need to be able to return to a previous state of the code
    (primary and any auxillary repositories must always be at specific revisions/releases) 


* local copies of table subsets into tmp_*username*_offline_db




