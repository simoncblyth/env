Investigating SVN 2 Git Migration
====================================

Need `git svn` in order to clone from SVN. 

 * http://git-scm.com/book/en/Git-and-Other-Systems-Migrating-to-Git


Prepare authors file
----------------------

Git associates commits with email addresses rather than user named like SVN.
So need to prepare a mapping file

Using trac report 11 is csv format 

  * http://dayabay.phys.ntu.edu.tw/tracs/env/report/11?format=csv 
 
  * :env:`trunk/scm/svnauthors.py`

::

   ~/env/scm/svnauthors.py read                    # reads the trac report 11 and inserts into sqlite3 DB
   ~/env/scm/svnauthors.py git > ~/svnusers.txt    # reads from DB, dumping in git author format



Trial Clone
---------------

::


    git svn clone http://dayabay.phys.ntu.edu.tw/repos/env/trunk --authors-file=~/svnusers.txt --no-metadata --stdlayout env

