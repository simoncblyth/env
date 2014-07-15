Migration Overview
===================

* https://confluence.atlassian.com/display/BITBUCKET/Convert+from+Subversion+to+Mercurial

code and revision history
---------------------------

#. see `hg-convert` trying hg convert 

   * http://mercurial.selenic.com/wiki/ConvertExtension
   * worked OK, 8 mins for env over network
   * timezone ?
   * authormap ?
   * how to verify systematically ?

#. alternative http://mercurial.selenic.com/wiki/HgSubversion

   * http://blogs.atlassian.com/2011/03/goodbye_subversion_hello_mercurial/

#. probably just leave: wiki markup in commit messages 

wiki
----

#. see `trac2bitbucket-wiki` 

   * needs a wiki repo first 

#. `tracwikidump.py` works, creates txt files for each Trac wikipage 

   * more of a backup than a migration 

tickets
--------

#. see `trac2bitbucket-tickets`

   * tickets have crazy dates from 1970, FIXED
   * http://www.redmine.org/issues/14567  


trac timestamps 
~~~~~~~~~~~~~~~~~~

Using Trac 0.11, so need to remove a factor of 1 million on timestamps.

In Trac API 0.12, the representation of timestamps was changed from seconds since the epoch
to microseconds since the epoch:

* http://trac.edgewall.org/wiki/TracDev/ApiChanges/0.12#Timestampstorageindatabase



