HG Convert of env repo from SVN to HG
=======================================

.. contents:: :local:


SVN env r1599 causing grief
------------------------------

::

    INFO:env.scm.migration.compare_hg_svn:compare_contents paths 499 svn_digest 499 hg_digest 499 mismatch 0  svn_only 0 hg_only 0 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1586 svnrev 1599 hgrev-svnrev -13 
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk rev 1599 to /tmp/subversion/env/ clean True 
    INFO:env.svn.bindings.svnclient:rmtree /tmp/subversion/env/ 
    Traceback (most recent call last):
    File "/Users/blyth/env/bin/compare_hg_svn.py", line 4, in <module>
        main()
    File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/scm/migration/compare_hg_svn.py", line 353, in main
        cf.compare(hgrev, svnrev)       
    File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/scm/migration/compare_hg_svn.py", line 244, in compare
        self.recurse(*hs)
    File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/scm/migration/compare_hg_svn.py", line 164, in recurse
        self.svn.recurse(svnrev)
    File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/svn/bindings/svnclient.py", line 146, in recurse
        self.checkout(rev, clean=clean)
    File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/svn/bindings/svnclient.py", line 135, in checkout
        self.client.checkout(self.url, self.path, revision=rev_(rev), ignore_externals=self.ignore_externals)
    pysvn._pysvn_2_7.ClientError: Revision 1599 doesn't match existing revision 1598 in '/tmp/subversion/env'
    (adm_env)delta:~ blyth$ 


SVN Tree Conflict
-------------------

Cannot navigate through this slice of history, get tree conflicts.
This means that the comparisons with the hg convert fail.

* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1599 1586  thho modifies trunk/thho/NuWa/python/histogram/pyhist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1600 1587  thho copies trunk/thho/NuWa/python/histogram/pyhist.py to trunk/thho/NuWa/python/histogram/PyHist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1601 1588  thho removes trunk/thho/NuWa/python/histogram/pyhist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1715       thho removes trunk/thho/NuWa/python


* http://blogs.collab.net/subversion/subversion-160-and-tree-conflicts#.U9oGC1YWe2w

Recall that Subversion performs renames as a combination of a copy and a delete.



Stepping through the HG converted history doesnt go into conflict state::

    delta:histogram blyth$ hg update -r1586
    0 files updated, 0 files merged, 1 files removed, 0 files unresolved
    delta:histogram blyth$ l
    total 16
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 16:32 pyhist_rename_to_avoid_degeneracy.py

    delta:histogram blyth$ hg update -r1587
    1 files updated, 0 files merged, 0 files removed, 0 files unresolved
    delta:histogram blyth$ l
    total 32
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 16:49 PyHist.py
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 16:32 pyhist_rename_to_avoid_degeneracy.py

    delta:histogram blyth$ hg update -r1588
    0 files updated, 0 files merged, 1 files removed, 0 files unresolved
    delta:histogram blyth$ l
    total 16
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 16:49 PyHist.py


Despite the apparent clean svnclient.py run tree conflicts are lurking:: 

    delta:subversion blyth$ svnclient.py -r1599,1600,1601
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk/ rev 1599 to /tmp/subversion/env/ 
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk/ rev 1600 to /tmp/subversion/env/ 
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk/ rev 1601 to /tmp/subversion/env/ 
    delta:subversion blyth$ 

    Despite the above there is lurking tree conflict

    delta:env blyth$ svn st 
    D     C thho/NuWa/python/histogram/PyHist.py
          >   local file unversioned, incoming file add upon update
    Summary of conflicts:
      Tree conflicts: 1
    delta:env blyth$ 


Workaround this pain, by doing clean checkouts for these revisions::

    (adm_env)delta:~ blyth$ svnclient.py -r1599 --clean-checkout-revs 1599,1600,1601
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk/ rev 1599 to /tmp/subversion/env/ clean True 
    INFO:env.svn.bindings.svnclient:rmtree /tmp/subversion/env/ 
    (adm_env)delta:~ blyth$ l /tmp/subversion/env/thho/NuWa/python/histogram/
    total 16
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 18:11 pyhist.py
    (adm_env)delta:~ blyth$ 
    (adm_env)delta:~ blyth$ svnclient.py -r1600 --clean-checkout-revs 1599,1600,1601
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk/ rev 1600 to /tmp/subversion/env/ clean True 
    INFO:env.svn.bindings.svnclient:rmtree /tmp/subversion/env/ 
    {'repos_lock': None, 'text_status': <wc_status_kind.missing>, 'repos_text_status': <wc_status_kind.none>, 'is_locked': 0, 'is_copied': 0, 'is_switched': 0, 'is_versioned': 1, 'prop_status': <wc_status_kind.none>, 'entry': <PysvnEntry u'PyHist.py'>, 'path': u'/tmp/subversion/env/thho/NuWa/python/histogram/PyHist.py', 'repos_prop_status': <wc_status_kind.none>}
    (adm_env)delta:~ blyth$ l /tmp/subversion/env/thho/NuWa/python/histogram/
    total 16
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 18:12 pyhist.py
    (adm_env)delta:~ blyth$ svn status  /tmp/subversion/env
    !       /tmp/subversion/env/thho/NuWa/python/histogram/PyHist.py
    (adm_env)delta:~ blyth$ 
    (adm_env)delta:~ blyth$ 
    (adm_env)delta:~ blyth$ svnclient.py -r1601 --clean-checkout-revs 1599,1600,1601
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk/ rev 1601 to /tmp/subversion/env/ clean True 
    INFO:env.svn.bindings.svnclient:rmtree /tmp/subversion/env/ 
    (adm_env)delta:~ blyth$ l /tmp/subversion/env/thho/NuWa/python/histogram/
    total 16
    -rw-r--r--  1 blyth  wheel  5258 Jul 31 18:13 PyHist.py
    (adm_env)delta:~ blyth$ 






::

    delta:env blyth$ st 
    !     C thho/NuWa/python/histogram/PyHist.py
          >   local file delete, incoming file delete upon update
    Summary of conflicts:
      Tree conflicts: 1
    delta:env blyth$ svn resolve --accept base thho/NuWa/python/histogram/PyHist.py
    svn: warning: W155027: Tree conflict can only be resolved to 'working' or 'mine-conflict' state; '/private/tmp/subversion/env/thho/NuWa/python/histogram/PyHist.py' not resolved
    svn: E205011: Failure occurred resolving one or more conflicts
    delta:env blyth$ 

    delta:env blyth$ svn resolve --accept mine-conflict thho/NuWa/python/histogram/PyHist.py





::

    delta:histogram blyth$ svn update -r1599
    Updating '.':
    Skipped 'PyHist.py' -- Node remains in conflict
    At revision 1599.
    Summary of conflicts:
      Skipped paths: 1

    delta:histogram blyth$ svn st 
    D     C PyHist.py
          >   local file unversioned, incoming file add upon update
    Summary of conflicts:
      Tree conflicts: 1


    delta:histogram blyth$ svn update -r1600
    Updating '.':
    At revision 1600.
    delta:histogram blyth$ 
    delta:histogram blyth$ ll
    total 16
    drwxr-xr-x  10 blyth  wheel   340 Jul 31 16:05 ..
    -rw-r--r--   1 blyth  wheel  5258 Jul 31 16:32 pyhist.py
    drwxr-xr-x   3 blyth  wheel   102 Jul 31 16:32 .
    delta:histogram blyth$ 
    delta:histogram blyth$ svn update -r1601
    Updating '.':
    D    pyhist.py
    Updated to revision 1601.
    delta:histogram blyth$ ll
    total 0
    drwxr-xr-x  10 blyth  wheel  340 Jul 31 16:05 ..
    drwxr-xr-x   2 blyth  wheel   68 Jul 31 16:56 .
    delta:histogram blyth$ svn update -r1602
    Updating '.':
    At revision 1602.
    delta:histogram blyth$ l
    delta:histogram blyth$ st 
    D     C PyHist.py
          >   local file unversioned, incoming file add upon update
    Summary of conflicts:
      Tree conflicts: 1
    delta:histogram blyth$ 









::

    INFO:env.scm.migration.compare_hg_svn:hgrev 1587 svnrev 1600 hgrev-svnrev -13 
    INFO:env.svn.bindings.svnclient:checkout file:///var/scm/subversion/env/trunk rev 1600 to /tmp/subversion/env/ 
    INFO:env.scm.migration.compare_hg_svn:1 ['hg_only_paths'] issues encountered in compare_paths
    lines_dirs

    lines_paths
     [l ] thho/NuWa/python/histogram/PyHist.py 
    Python 2.7.8 (default, Jul 13 2014, 17:11:32) 
    Type "copyright", "credits" or "license" for more information.

    IPython 2.1.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: hg_only_paths
    Out[1]: ['thho/NuWa/python/histogram/PyHist.py']


::

    delta:histogram blyth$ ll /tmp/{mercurial,subversion}/env/thho/NuWa/python/histogram
    /tmp/subversion/env/thho/NuWa/python/histogram:
    total 16
    drwxr-xr-x  10 blyth  wheel   340 Jul 31 16:05 ..
    -rw-r--r--   1 blyth  wheel  5258 Jul 31 16:32 pyhist.py
    drwxr-xr-x   3 blyth  wheel   102 Jul 31 16:32 .

    /tmp/mercurial/env/thho/NuWa/python/histogram:
    total 32
    drwxr-xr-x  10 blyth  wheel   340 Jul 31 15:33 ..
    -rw-r--r--   1 blyth  wheel  5258 Jul 31 16:32 pyhist_rename_to_avoid_degeneracy.py
    -rw-r--r--   1 blyth  wheel  5258 Jul 31 16:32 PyHist.py
    drwxr-xr-x   4 blyth  wheel   136 Jul 31 16:32 .
    delta:histogram blyth$ 



[RESOLVED] Degeneracy Back again
-----------------------------------

Leading slash in paths prevent the filemap renames being applied. Resolved by 
ensuring recursion roots have a trailing slash, hence ensuring root relatives
to not have a leading slash. 

::

    INFO:env.scm.migration.compare_hg_svn:hgrev 1583 svnrev 1596 hgrev-svnrev -13 
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk rev 1596 to /tmp/subversion/env 
    INFO:env.scm.migration.compare_hg_svn:1 ['hg_only_paths'] issues encountered in compare_paths
    lines_dirs

    lines_paths
     [ r] /thho/NuWa/python/histogram/pyhist.py 
     [l ] /thho/NuWa/python/histogram/pyhist_rename_to_avoid_degeneracy.py 
    INFO:env.scm.migration.compare_hg_svn:compare_contents paths 499 svn_digest 500 hg_digest 500 mismatch 0  svn_only 1 hg_only 1 
    INFO:env.scm.migration.compare_hg_svn:issues encountered in compare_contents : [('hg_keys', False), ('svn_only', False), ('hg_only', False)] 
    Python 2.7.8 (default, Jul 13 2014, 17:11:32) 
    Type "copyright", "credits" or "license" for more information.

    IPython 2.1.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: 




And this directory link ?
---------------------------

::

     http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/qxml
     link db/bdbxml/qxml

     http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/db/bdbxml/qxml/



[RESOLVED] Credentials callback from r731 
-------------------------------------------

Attempts to checkout revisions from 731 need credentials. 
Resolved by ignoring externals in the pysvn checkout

* http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/dyb/NuWa?rev=731  svn:externals testing 


::

    (adm_env)delta:~ blyth$ svnclient.py -r732
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 732 to /tmp/subversion/env 
    INFO:env.svn.bindings.svnclient:get_login realm <http://dayabay.ihep.ac.cn:80> svn-repos username blyth may_save 1 
    ValueError: invalid literal for int() with base 10: ''
    Traceback (most recent call last):
      File "/Users/blyth/env/bin/svnclient.py", line 4, in <module>
        main()
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/svn/bindings/svnclient.py", line 259, in main
        sc.recurse(rev)
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/svn/bindings/svnclient.py", line 121, in recurse
        self.checkout(rev)
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/svn/bindings/svnclient.py", line 117, in checkout
        self.client.checkout(self.url, self.path, revision=rev_(rev))
    pysvn._pysvn_2_7.ClientError: unhandled exception in callback_get_login
    (adm_env)delta:~ blyth$ 

::

    delta:NuWa blyth$ svn pg svn:externals
    dybgaudi -r1863 http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/branches/releases/0.2.0
    gaudi -r1864 http://dayabay.ihep.ac.cn/svn/dybsvn/gaudi/branches/releases/0.2.0
    lcgcmt -r1867 http://dayabay.ihep.ac.cn/svn/dybsvn/lcgcmt/branches/releases/0.2.0
    ldm -r1868 http://dayabay.ihep.ac.cn/svn/dybsvn/ldm/branches/releases/0.0.3
    tut_anal1 -r1869 http://dayabay.ihep.ac.cn/svn/dybsvn/tut_anal1/branches/releases/0.0.1
    installation http://dayabay.ihep.ac.cn/svn/dybsvn/installation/branches/inst-NuWa-0.0.4/dybinst




[RESOLVED] What is special about this stretch of SVN history ?   
---------------------------------------------------------------

Creation of a folder than contains nothing but other empties was not 
skipped by  `--skipempty` in SVNCrawler.  Resolved by judging emptiness at the
tail rather than head of the recursion based on the total leaves beneath a node.

Discrepant folder `/thho/NuWa` despite `--skipempty` enables for the SVN crawl::

    INFO:env.scm.migration.compare_hg_svn:hgrev 1433 svnrev 1445 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1434 svnrev 1447 
    INFO:env.scm.migration.compare_hg_svn:1 ['svn_only_dirs'] issues encountered in compare_paths
    lines_dirs
     [ r] /thho/NuWa           
    lines_paths

    INFO:env.scm.migration.compare_hg_svn:hgrev 1435 svnrev 1448 
    INFO:env.scm.migration.compare_hg_svn:1 ['svn_only_dirs'] issues encountered in compare_paths
    lines_dirs
     [ r] /thho/NuWa           
    lines_paths

    INFO:env.scm.migration.compare_hg_svn:hgrev 1436 svnrev 1449 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1437 svnrev 1450 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1438 svnrev 1451 


* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1446  creation of "NuWa" directory than contains only an empty "python" directory `trunk/thho/NuWa/python`
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1447  unrelated 
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1448  unrelated
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1449  thho populates 2 levels of emptyness, by creating `trunk/thho/NuWa/python/gentools.py`


Maybe resolved too ?
-----------------------

Even with skipempty, this is still tripping up:: 

        INFO:env.scm.migration.compare_hg_svn:hgrev 644 svnrev 646 
        lines_dirs
         [ r] /thho                
        lines_paths


Directory symbolic links need separate handling 
-------------------------------------------------

This is a problem with the comparison, the conversion succeeds to 
translate the SVN directory link into a Mercurial one. 

::

    delta:env blyth$ pwd
    /tmp/mercurial/env
    delta:env blyth$ l qxml*
    lrwxr-xr-x  1 blyth  wheel  14 Jul 30 16:28 qxml -> db/bdbxml/qxml

    delta:e blyth$ pwd
    /Users/blyth/e
    delta:e blyth$ l qxml
    lrwxr-xr-x  1 blyth  staff  14 Jan 14  2014 qxml -> db/bdbxml/qxml



::

    In [3]: svn_only_paths
    Out[3]: ['/qxml']



::

    INFO:env.scm.migration.compare_hg_svn:hgrev 3470 svnrev 3493 
    INFO:env.scm.migration.compare_hg_svn:2 ['hg_only_paths', 'hg_only_dirs'] issues encountered in compare_paths
    lines_dirs
     [l ] /qxml/test           
     [l ] /qxml                
    lines_paths
     [l ] /qxml/common.cc      
     [l ] /qxml/config.hh      
     [l ] /qxml/existmeta.py   
     [l ] /qxml/extresolve.hh  
     [l ] /qxml/extfun.py      
     [l ] /qxml/common.py      
     [l ] /qxml/potools.cc     
     [l ] /qxml/extfun.cc      
     [l ] /qxml/qxmlcfg.cc     
     [l ] /qxml/test/tpy.xq    
     [l ] /qxml/makeXmlException.inc 
     [l ] /qxml/config.py      
     [l ] /qxml/element.cc     
     [l ] /qxml/extfun.hh      
     [l ] /qxml/Makefile       
     [l ] /qxml/extresolve.cc  
     [l ] /qxml/extfun.i       
     [l ] /qxml/qxml.py        
     [l ] /qxml/monolith.py    
     [l ] /qxml/potools.hh     
     [l ] /qxml/model.hh       
     [l ] /qxml/qxml.cc        
     [l ] /qxml/hfagc.cfg      
     [l ] /qxml/exist2qxml.py  
     [l ] /qxml/test_pyextfun.py 
     [l ] /qxml/quote.py       
     [l ] /qxml/element.hh     
     [l ] /qxml/transfer.py    
     [l ] /qxml/hfagc.dbxml    
     [ r] /qxml                
     [l ] /qxml/throwPyUserException.inc 
     [l ] /qxml/glyph.py       
     [l ] /qxml/README.txt     
     [l ] /qxml/test/extmixed.xq 
     [l ] /qxml/notes.txt      
     [l ] /qxml/config.cc      
     [l ] /qxml/test/ext.xq    
     [l ] /qxml/setup.py       
     [l ] /qxml/common.hh      
     [l ] /qxml/test/tpydump.xq 
     [l ] /qxml/test/ls.xq     
     [l ] /qxml/model.cc       
    INFO:env.scm.migration.compare_hg_svn:issues encountered in compare_contents
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    Type "copyright", "credits" or "license" for more information.



[RESOLVED] Comparison needs to apply the filemap 
---------------------------------------------------

Resolved by application of `hg convert` filemap renames 
to SVN paths before comparison with the HG paths. The renames
were needed in the first place to avoid case folding problem.

::

    INFO:env.scm.migration.compare_hg_svn:hgrev 1583 svnrev 1596 
    INFO:env.scm.migration.compare_hg_svn:2 ['hg_only_paths', 'svn_only_paths'] issues encountered in compare_paths
    lines_dirs

    lines_paths
     [ r] /thho/NuWa/python/histogram/pyhist.py 
     [l ] /thho/NuWa/python/histogram/pyhist_rename_to_avoid_degeneracy.py 
    INFO:env.scm.migration.compare_hg_svn:issues encountered in compare_contents
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    Type "copyright", "credits" or "license" for more information.


Dud svn rev 10
----------------------------

While restructing to trunk the following offsets are seen.  Observing early offsets of 3 when doing fullrepo.

::

    (adm_env)delta:~ blyth$ compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1:10 --hgrev 0:9  
    INFO:env.scm.migration.compare_hg_svn:hgrev 0 svnrev 1 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1 svnrev 2 
    INFO:env.scm.migration.compare_hg_svn:hgrev 2 svnrev 3 
    INFO:env.scm.migration.compare_hg_svn:hgrev 3 svnrev 4 
    INFO:env.scm.migration.compare_hg_svn:hgrev 4 svnrev 5 
    INFO:env.scm.migration.compare_hg_svn:hgrev 5 svnrev 6 
    INFO:env.scm.migration.compare_hg_svn:hgrev 6 svnrev 7 
    INFO:env.scm.migration.compare_hg_svn:hgrev 7 svnrev 8 
    INFO:env.scm.migration.compare_hg_svn:hgrev 8 svnrev 9       ## offset of 1 due to trunk restriction, up to dud svn rev 10

    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 9 

    (adm_env)delta:~ blyth$ compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10:19 --hgrev 8:17 
    INFO:env.scm.migration.compare_hg_svn:hgrev 8 svnrev 10 
    INFO:env.scm.migration.compare_hg_svn:hgrev 9 svnrev 11 
    INFO:env.scm.migration.compare_hg_svn:hgrev 10 svnrev 12 
    INFO:env.scm.migration.compare_hg_svn:hgrev 11 svnrev 13   
    INFO:env.scm.migration.compare_hg_svn:hgrev 12 svnrev 14 
    INFO:env.scm.migration.compare_hg_svn:hgrev 13 svnrev 15 
    INFO:env.scm.migration.compare_hg_svn:hgrev 14 svnrev 16 
    INFO:env.scm.migration.compare_hg_svn:hgrev 15 svnrev 17  
    INFO:env.scm.migration.compare_hg_svn:hgrev 16 svnrev 18       ## beyond the dud, need offset of 2 to match

    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 -A 

    INFO:env.scm.migration.compare_hg_svn:hgrev 388 svnrev 390 
    lines_dirs
     [ r] /seed                
    lines_paths


Other issues
--------------

#. empty folders
#. symbolic links 


[RESOLVED] Case folding collision
--------------------------------------

SVN permits case degenerate paths to have distinct entries in its DB, but Mercurial doesnt.
Resolved, by filemap rename within the hg convert to avoid the degeneracy ever happening.

Problematic bits of history:

* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1599   thho modifies trunk/thho/NuWa/python/histogram/pyhist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1600   thho copies trunk/thho/NuWa/python/histogram/pyhist.py to trunk/thho/NuWa/python/histogram/PyHist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1601   thho removes trunk/thho/NuWa/python/histogram/pyhist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1715   thho removes trunk/thho/NuWa/python


* http://dayabay.phys.ntu.edu.tw/tracs/env/log/trunk/thho/NuWa/python/histogram?rev=1714


The update to (hgrev 1587 svnrev 1600) gives case-folding collision still (filemap rename not working?)::


    INFO:env.scm.migration.compare_hg_svn:hgrev 1586 svnrev 1599 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1587 svnrev 1600 
    Traceback (most recent call last):
      File "/Users/blyth/env/bin/compare_hg_svn.py", line 4, in <module>
        main()
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/scm/migration/compare_hg_svn.py", line 318, in main
        hg.recurse(hgrev)   # updates hg working copy to this revision
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/env/hg/bindings/hgcrawl.py", line 216, in recurse
        self.hg.hg_update(hgrev)
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/hgapi/hgapi.py", line 173, in hg_update
        self.hg_command(*cmd)
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/hgapi/hgapi.py", line 113, in hg_command
        return Repo.command(self.path, self._env, *args)
      File "/usr/local/env/adm_env/lib/python2.7/site-packages/hgapi/hgapi.py", line 95, in command
        exit_code=proc.returncode)
    hgapi.hgapi.HgException: Error running hg --cwd /tmp/mercurial/env update 1587:
    " + tErr: abort: case-folding collision between thho/NuWa/python/histogram/pyhist.py and thho/NuWa/python/histogram/PyHist.py

        Out: 
        Exit: 255


::

    (adm_env)delta:env blyth$ hg update -r1586
    251 files updated, 0 files merged, 2641 files removed, 0 files unresolved


filemap not working without the trunk::

    (adm_env)delta:env blyth$ hg update -r1586
    251 files updated, 0 files merged, 2641 files removed, 0 files unresolved
    (adm_env)delta:env blyth$ 
    (adm_env)delta:env blyth$ 
    (adm_env)delta:env blyth$ cd thho/NuWa/python/histogram/
    (adm_env)delta:histogram blyth$ l
    total 16
    -rw-r--r--  1 blyth  wheel  5258 Jul 29 20:44 pyhist.py
    (adm_env)delta:histogram blyth$ pwd
    /tmp/mercurial/env/thho/NuWa/python/histogram
    (adm_env)delta:histogram blyth$ 

    (adm_env)delta:histogram blyth$ hg update -r1587
    abort: case-folding collision between thho/NuWa/python/histogram/pyhist.py and thho/NuWa/python/histogram/PyHist.py


Argh case degenerate entries at SVN rev 1600::

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1599 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/pyhist.py

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1600 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/PyHist.py
    /trunk/thho/NuWa/python/histogram/pyhist.py

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1601 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/PyHist.py

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1602 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/PyHist.py



hg only
----------

The hg repo is created from original SVN repo via network. The snapshot of SVN repo
comes from a backup. This explains these contiguous recent commits being only in hg.

::

    In [22]: for _ in sorted(ho):print "%(hrev)s %(log)s " % hh[_]

    4608 looking into trac migration into mercurial, comparing checkout from hg converted repo to original svn working copy 
    4609 eliminating empty directories by deletion or adding empty README.txt as cause problems for comparison with mercurial migrated repo checkouts 
    4610 a few more empty dirs, now with README 
    4611 hg convert testing, so need to keep getting SVN to clean revisions 
    4612 more svn/hg diffs 
    4613 env working copy between svn and hg converted almost perfect match now 
    4614 mercurial notes 
    4615 comparing env hg/svn history, find dud revision 10 
    4616 machinery for new virtualenv adm- python, for sysadmin tasks like migarted to mercurial vs svn history comparisons 
    4617 generalize tracmigrate into scmmigrate, investigate hgapi and svn bindings 
    4618 svn and hg crawlers now check directory correspondence between revisions, not yet content 
    4619 extend hg and svn crawlers to compare file content at all revisions, fix issues with symbolic links, problem of case degeneracy remains 


svn only
----------

Manual log check doing::

    delta:e blyth$ svn log -r4000 -v 
    ------------------------------------------------------------------------
    r4000 | lint | 2013-10-22 13:24:09 +0800 (Tue, 22 Oct 2013) | 1 line
    Changed paths:
       M /trunk/lintao/archive

    add the latest directory.


Indicates SVN onlys are caused by  

#. creations/deletions of empty directories
#. dud revision 10
#. svn property changes



::


    In [24]: for _ in sorted(so):print "%(srev)s %(log)s " % ss[_]
    1 initial import from dummy  
    10 
    delete swp
     
    390 for random seed, hostid checking
     
    646 thho work area 
    729 tidy up unused folders 
    730 tidy 
    731 svn:externals testing  
    738 avoid slow NuWa update on every env-u !  
    1264 acrylic sample study 
    1443 try to move some if the bitten setup into the server rather than the checkout scripts with svn:externals  
    1444 switch order to workaround ... ''SSL is not supported'' in the bitten checkout  
    1446 NuWa tips 
    ...

