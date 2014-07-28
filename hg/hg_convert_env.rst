HG Convert of env repo from SVN to HG
=======================================

log comparison
----------------

#. Need to normalize the message. 
#. Adopt alignment based on timestamp, for easy debug of whats causing the skips

::


    In [2]: ["%(hrev)s %(log)s " % hg.log[_] for _ in range(0,10)]
    Out[2]: 
    [u'0 initial scm-import ',
     u'1 tidy up ',
     u'2 tweaks ',
     u'3 allow ttho to get NODE_TAG of G1 ',
     u'4 new funcs allowing apache2 to run as a service ... providing auto restart\nafter powercuts etc.. ',
     u'5 remove swp ',
     u'6 service setup... ',
     u'7 tweak ',
     u'8 finally got apache2 working as a service... we shall see\nat the next powercut if trac+svn bounces back automatically ',
     u'9 xmlrpc plugin ']

    In [5]: ["%(srev)s %(log)s " % svn.log[_] for _ in range(0,13)]
    Out[5]: 
    ['0 None ',                             ### not converted
     '1 initial import from dummy  ',       ### not converted
     '2 initial scm-import  ',              ### offset +2  
     '3  \n tidy up \n ',
     '4 tweaks\n ',
     '5 allow ttho to get NODE_TAG of G1\n ',
     '6 \nnew funcs allowing apache2 to run as a service ... providing auto restart\nafter powercuts etc..\n\n\n ',
     '7 \nremove swp\n\n ',
     '8 \nservice setup... \n\n ',
     '9 tweak\n ',
     '10 \ndelete swp\n ',              #### dud commit 10 ?
     '11 \nfinally got apache2 working as a service... we shall see\nat the next powercut if trac+svn bounces back automatically\n\n ',  ### offset +3
     '12 xmlrpc plugin ']   


    In [6]: ["%(srev)s %(log)s " % svn.log[_] for _ in range(100,100+12)]
    Out[6]: 
    ['100 attempt condor wo shared filesystem for outputs ',
     '101 attempt condor wo shared filesystem for outputs ',
     '102 attempt condor wo shared filesystem for outputs ',
     '103 attempt condor wo shared filesystem for outputs ',
     '104 refactor to use condor-lookup , avoiding duplicated strings ',
     '105 refactor to use condor-lookup , avoiding duplicated strings ',
     '106 env-u to update and source  ',
     '107 env-u document ',
     '108 abort condor non-shared attempt ... set OUTPUT_BASE\nto USER_BASE ',
     '109 add /jobs ',
     '110 path fixes\n ',
     '111 path fixes\n ']

    In [7]: ["%(hrev)s %(log)s " % hg.log[_] for _ in range(100-3,100+12-3)]
    Out[7]: 
    [u'97 attempt condor wo shared filesystem for outputs ',
     u'98 attempt condor wo shared filesystem for outputs ',
     u'99 attempt condor wo shared filesystem for outputs ',
     u'100 attempt condor wo shared filesystem for outputs ',
     u'101 refactor to use condor-lookup , avoiding duplicated strings ',
     u'102 refactor to use condor-lookup , avoiding duplicated strings ',
     u'103 env-u to update and source ',
     u'104 env-u document ',
     u'105 abort condor non-shared attempt ... set OUTPUT_BASE\nto USER_BASE ',
     u'106 add /jobs ',
     u'107 path fixes ',
     u'108 path fixes ']






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


Case folding collision
------------------------

SVN permits case degenerate paths to have distinct entries in its DB, but Mercurial doesnt.

Problematic bits of history:

* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1599   thho modifies trunk/thho/NuWa/python/histogram/pyhist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1600   thho copies trunk/thho/NuWa/python/histogram/pyhist.py to trunk/thho/NuWa/python/histogram/PyHist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1601   thho removes trunk/thho/NuWa/python/histogram/pyhist.py
* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/1715   thho removes trunk/thho/NuWa/python


#. attempting to tackle this via hg convert filemap, to prevent the degeneracy ever getting into Mercurial

::

    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 -A  --skipempty 


    INFO:env.scm.migration.compare_hg_svn:hgrev 1598 svnrev 1600 
    ---------------------------------------------------------------------------
    HgException                               Traceback (most recent call last)
    ...
    HgException: Error running hg --cwd /tmp/mercurial/env update 1598:
    " + tErr: abort: case-folding collision between thho/NuWa/python/histogram/pyhist.py and thho/NuWa/python/histogram/PyHist.py


    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1600 --hgrev 1598 -A  --skipempty 

        ## keep getting this...


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


