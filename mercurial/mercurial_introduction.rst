Mercurial
===========

overview
--------

#. mercurial encourages one-branch-per-repo pattern of usage

   * nice simplicity, just use clone to "branch" 
   * http://hgbook.red-bean.com/read/managing-releases-and-branchy-development.html#id386031

hg book
-------

* http://hgbook.red-bean.com
* http://hgbook.red-bean.com/read/
* http://hgbook.red-bean.com/read/migrating-to-mercurial.html


Subversion brain damage remediation
-------------------------------------

* http://hginit.com/00.html


Mecurial Approach
-------------------

* commands apply to the entire repo not just a subdir 
* branch by cloning into a separate repo, 
  can merge that back to a central repo later (maybe)


hg vs git branching
---------------------

* http://mercurial.selenic.com/wiki/BranchingExplained
* http://stevelosh.com/blog/2009/08/a-guide-to-branching-in-mercurial/


hg branches
-------------

::

    (chroma_env)delta:chroma blyth$ hg branches
    default                      232:b565b38ae23a
    virtual_triangle             157:4cfe4b4ca7fb
    weighting                     67:3301d3b2bc44
    no-bootstrap                 230:3a3d0228c43a (inactive)
    server                       201:494c7b2b742c (inactive)
    newcore                      122:9a1996f130ed (inactive)
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ pwd
    /usr/local/env/chroma_env/src/chroma/chroma
    (chroma_env)delta:chroma blyth$ 


hg basics
----------

* http://mercurial.selenic.com/wiki/QuickStart


setup `~/.hgrc`
~~~~~~~~~~~~~~~~


summary
~~~~~~~~

::

    (chroma_env)delta:chroma blyth$ hg sum
    parent: 232:b565b38ae23a tip
     remove print statement from setup.py
    branch: default
    commit: 1 modified
    update: (current)
    (chroma_env)delta:chroma blyth$ 


status/commit/push
~~~~~~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:chroma blyth$ hg status
    M setup.py
    (chroma_env)delta:chroma blyth$ hg commit -m "import ROOT into setup.py avoids atexit cleanup error when running: python setup.py test" 
    (chroma_env)delta:chroma blyth$ hg status
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/scb-/chroma
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 1 changesets with 1 changes to 1 files
    (chroma_env)delta:chroma blyth$ 


Commit is visible in web interface

* https://bitbucket.org/scb-/chroma





