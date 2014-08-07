Migration Overview
===================

* https://confluence.atlassian.com/display/BITBUCKET/Convert+from+Subversion+to+Mercurial

2-stage strategy
------------------

Attack in two stages:

#. migrate "env" asis into bitbucket


Split
--------

Not really a migration, this is new creation.

#. create a (3?) new repositories with 

   * g4dae : just the geant4 exporter 
   * g4daeview : visualization, optional dependency on g4daechroma
   * g4daechroma : mapping from g4dae into chroma and propagation steering 
 
Division to allow users without Chroma/CUDA to still benefit from
the visualization functionality.


* `hg convert` can extract history of parts of the SVN repo in
  its conversion into Mercurial, so can partition into
  separate repos.  Which can then be fixed up for the different 
  python import paths. 



code and revision history
---------------------------

#. see `hg-convert` trying hg convert 

   * http://mercurial.selenic.com/wiki/ConvertExtension
   * worked OK, 8 mins for env over network
   * timezone ?
   * authormap ?
   * how to verify systematically ?

#. alternative http://mercurial.selenic.com/wiki/HgSubversion

   * TODO: check this recommendation of bitbucket  http://blogs.atlassian.com/2011/03/goodbye_subversion_hello_mercurial/

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



