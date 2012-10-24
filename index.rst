:modified: 2012-07-10 08:42:54+00:00
:tags: Red,Green,Blue

Env Development Log
=======================

:Version: |version|
:Date: |today|


Objectives for these pages.

#. TODO list  
#. development log

Major issues should continue to be tracked via env trac tickets 
but often a large number of smaller issues are dealt with when performing recoveries.  
The Trac interface is suited to long lived issues, whereas a more logging style of
interface is useful to capture operational fixes and techniques.  

Have formerly been using Trac wiki pages for this purpose, but that looses the 
intimate context of having sources of documentation residing in the same 
repository as the code being developed, and suffers from inconvenient 
web interface text editing.

Seeing **TODO** items and logged fixes in the same commits as the 
code changes is invaluable.

Very structured information closely linked to bash functions should be logged
within the bash functions themselves. For example:


Contents:

.. toctree::
   :maxdepth: 3

   TODO <TODO>
   LOG <log/May2012>
   Sys Admin <sysadmin/index>
   Plotting <plot/index>
   SCM <scm/index>
   ROOT <root/index>
   ENV <_docs/env>
   sphinxext/index
   matplotlib/index   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

