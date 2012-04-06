
Issues/Enhancements/Ideas
==========================

* avoid absolute paths in config file 
   * maybe allow envvar interpolation for a list of named envvars, eg HEPREZ_HOME QXML_TMP  

* ``dbxml`` shell like capabilities for ``qxml`` that hook into the qxml configuration  
   /usr/local/env/db/dbxml-2.5.16/dbxml/src/utils/shell/dbxmlsh.cpp

* resolver rationalization
   * python resolver / C++ resolver / swigged C++ resolver 
   * python resolver palm off to swigged C++ resolver ? 
   * separate ns for python and swigged C++ for implementation comparisons

* query timing reporting 

* configuration of indices



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


Using C++ API
==============

.. warning:: DB XML docs have mismatches to header signatures, **trust headers above documentation**


Using dbxml shell
===================


A script containg dbxml commands can save some typing::

	cat  hfagc.dbxml

	openContainer /tmp/hfagc/hfagc_system.dbxml
	addAlias sys
	openContainer /tmp/hfagc/hfagc.dbxml
	addAlias hfc

::

	simon:qxml blyth$ dbxml -h /tmp/dbxml -s hfagc.dbxml
	Joined existing environment

	dbxml> query 'collection("dbxml:/sys")'
	stdin:1: query failed, Error: Cannot resolve container: sys.  Container not open and auto-open is not enabled.  Container may not exist.

	dbxml>  query 'collection("dbxml:/hfc")'
	226 objects returned for eager expression 'collection("dbxml:/hfc")'





Observations on Berkeley DB XML XQuerying
==========================================

#. ``document-uri(root($smth))``  does not work, in more involved locations .. suspect a steps removed effect
     dbxml:metadata("dbxml:name",$smth)  seems to work OK without need to ``root`` up to the doc.




