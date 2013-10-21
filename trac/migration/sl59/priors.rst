Prior Migration Notes
=======================

.. contents:: :local:


Documentation Sources
----------------------

ENV repo .rst sources and Sphinx derived html

   * http://dayabay.phys.ntu.edu.tw/repos/env/trunk/
   * http://dayabay.phys.ntu.edu.tw/e/  

Admin Operating Procedures

   * http://belle7.nuu.edu.tw/oum/aop/
   * http://dayabay.bnl.gov/oum/aop/   (currently not being updated due to build issue)
   * http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/Documentation/OfflineUserManual/tex/aop/   (.rst sources in dybsvn)

ENV and DYBSVN repo Trac wikis

   * http://dayabay.phys.ntu.edu.tw/tracs/env
   * http://dayabay.ihep.ac.cn/tracs/dybsvn

Official Trac pages

   * http://trac.edgewall.org/wiki/0.11/TracInstall


ENV Trac Wiki
--------------

Prior migrations were some years ago. So most docs likely to be 
in ENV Trac wiki rather than the more recent era RST/Sphinx docs.

Wikiology  
~~~~~~~~~~~

Mining Trac tagged wikipages

* http://dayabay.phys.ntu.edu.tw/tracs/env/tags?q=%27Trac%27 

* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/TracFromScratch   (Aug 2008)
* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/CMS02TracFromScratch  (Mar 2009)
* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/CMS01Recover  (Mar 2009)
* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/Belle7TracFromScratch (Mar/Apr 2009)
* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/CMS02TracFromScratchAgain  (end Apr 2009)   

  * introduced toplevel `trac-build`

* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/Dyb01Trac  (Mar 2009) (debug communication between Simon and Qiumei)
* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/TracWithSystemApache  (April 2009)
* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/TracSVNMigrationJuly2009  (July 2009)

  * forensics on an uncontrolled migration gone awry

* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/Dyb02Trac (July 2009)

  * the **Dyb02Trac** page looks to be **VERY RELEVANT** to current task 
  * system prerequiste checking with yum, introduced `tracpreq-mode` of **system**



tracpreq
^^^^^^^^^^

Does source builds of python/swig/apache/svn/sqlite !

Hopefully the system/distro versions of these from SL59 work with Trac 0.11 
avoiding the need to build all these from source.  Need to ensure `tracpreq-mode` is **system**

::

   trac-
   tracpreq-         
   tracpreq-again  


tracbuild
^^^^^^^^^

::

   trac-
   tracbuild-
   tracbuild-auto

















