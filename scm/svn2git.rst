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

Git SVN Clone
---------------

About 100 revisions per minute on G, so will take at least 40min

::

    simon:wc blyth$ cp ~/svnusers.txt .
    simon:wc blyth$ time git svn clone http://dayabay.phys.ntu.edu.tw/repos/env/ --authors-file=svnusers.txt --no-metadata --stdlayout env
    Initialized empty Git repository in /private/tmp/wc/env/.git/
    r1 = 90e68c08508794f4aac96b24f6c82cd77751fb24 (refs/remotes/trunk)
            A       scm/modwsgi-use.bash
            A       scm/apache2.bash
            A       scm/svn-build.bash
              ....
            A       base/base.bash
            A       base/local.bash
            A       base/ssh.bash
            A       base/tty.bash
            A       base/perl.bash
    r2 = f8de820982167dcb3643c2ace9cb0c6a71c73b3a (refs/remotes/trunk)
            D       scm/.scm-use.bash.swp
            M       scm/trac-use.bash
            M       env.bash
            A       TODO
    W: -empty_dir: trunk/scm/.scm-use.bash.swp
    r3 = 4ea572aa7f612e4c64c4454d40cb885d6aa8b7ae (refs/remotes/trunk)
            M       env.bash
            M       TODO
    r4 = 8bab265e5aaf8cca91731ea7fc62ee9cbc32d222 (refs/remotes/trunk)
            M       base/local.bash
    r5 = d41c3f365350ea508b3301d80db6333070f12cde (refs/remotes/trunk)
            M       scm/apache2.bash


