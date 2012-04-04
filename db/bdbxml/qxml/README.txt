
Steps to add a C++ extension function
=======================================

#. implement in ``extfun.{cc,hh}`` 
#. add argument signature to ``extresolve.cc``
#. build C++ qxml with ``make``
#. add test calls to ``test/ext.xq`` that exercise the extension



Steps to make C++ extension available from python main XQueries
===============================================================

#. setup swig wrapping in ``extfun.i``



Using the python API
=====================

Very close to C++, but not the same need to examine::

    vi /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/dbxml.py



Ideas
======

* "dbxml" shell like capabilities for "qxml" that hook into the qxml configuration  
   /usr/local/env/db/dbxml-2.5.16/dbxml/src/utils/shell/dbxmlsh.cpp



