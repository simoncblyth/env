
Steps to add a C++ extension function
=======================================

#. implement in ``extfun.{cc,hh}`` 
#. add argument signature to ``extresolve.cc``
#. build C++ qxml with ``make``
#. add test calls to ``test/ext.xq`` that exercise the extension



Steps to make C++ extension available from python main XQueries
===============================================================

#. setup swig wrapping in ``extfun.i``


