
STDIN query piping and shebang line running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage examples assuming bash shell.
Piping from echo (must escape some chars from shell, handy for one-liners)::

     echo collection\(\)[1]               | qxml -
     echo "collection()[1]//rez:quote[1]" | qxml -
     echo "count(collection())"           | qxml -
     echo "count(collection()/rez:rez)"   | qxml -

NB use of configured default container avoids::

    echo "collection('dbxml:/hfc')[1]"   | qxml -
    echo "collection('dbxml:/tmp')[1]"   | qxml -
          ## containers identified by configured aliases or tags

    echo "collection('dbxml:////tmp/hfagc/hfagc.dbxml')/dbxml:metadata('dbxml:name')" | qxml -  
          ## explicit file path to the container


Useful for quick syntax checking::

     echo "let \$a := (1,2,3,4,5) return \$a[last()] " | qxml -
     echo 'let  $a := (1,2,3,4,5) return  $a[last()] ' | qxml -
     echo 'let $a := (1,2,3,4,5) return $a[position() <= 3] ' | qxml -

Grabbing a resource by name::

     echo "collection()/*[dbxml:metadata('dbxml:name')='/cdf/cjl/cdf_summer2007_BsDsK.xml']" | qxml -
     echo "collection()/*[dbxml:metadata('dbxml:name')='/cdf/cjl/cdf_summer2007_BsDsK.xml']" | qxml - > out.xml
          # redirect stdout to file
     echo "collection()/*[dbxml:metadata('dbxml:name')='/cdf/cjl/cdf_summer2007_BsDsK.xml']" | qxml - -o cdf.xml
          # writing into configured container with DB
  
Here strings (again must escape)::

     qxml - <<< collection\(\)[1]
     qxml - <<< "collection()[1]"

Here documents, must **not** do any escaping (also handy for few-liners)::

     qxml - << EOQ
     > collection()[1]
     > EOQ

From a bash function (uses another function to format arguments as an XQuery sequence)::

     rezlatex-code2latex-(){ qxml - << EOQ
     import module namespace my="http://my" at "my.xqm" ; 
     my:code2latex($(rezlatex-xqseq $*)) 
     EOQ
     }

Start turning command into script::

     cat - << EOQ > demo.xq
     > collection()[1]
     > EOQ
     cat demo.xq | qxml -

Shebang line running::

     cat - << EOQ > script.xq 
     #!/usr/bin/env qxml
     collection()[1]
     EOQ

     chmod ugo+x script.xq
     ./script.xq

Quick module import and invoke::

     echo 'import module namespace my="http://my" at "my.xqm" ; my:metadata(collection()[100]) ' | qxml -
     echo 'import module namespace my="http://my" at "my.xqm" ; my:code2latex("211") ' | qxml -
     # 
     # CONSIDERING configurable xquery module import prolog invoked via module command line option
     #  
     #              echo 'my:code2latex("211")' | qxml - -m my -m rz    
     #
     #  Simple way of doing this would offset error line numbers, but can pony up imports on 1st line, 
     #  register module library tags such as "my" in the config. 
     #

Issues/Enhancements/Ideas
==========================

* avoid absolute paths in config file 
   * maybe allow envvar interpolation for a list of named envvars, eg HEPREZ_HOME QXML_TMP 
      * use python style eg  %(HEPREZ_HOME)s/some/path  
           * makes python implementation easy  
           * cpp easier than shell style $HEPREZ_HOME/some/path

* ``dbxml`` shell like capabilities for ``qxml`` that hook into the qxml configuration  
   /usr/local/env/db/dbxml-2.5.16/dbxml/src/utils/shell/dbxmlsh.cpp

* resolver rationalization
   * python resolver / C++ resolver / swigged C++ resolver 
   * python resolver palm off to swigged C++ resolver ? 
   * separate ns for python and swigged C++ for implementation comparisons

* on writing xml into dbxml containers fill in created/modified/owner metadata

* configuration of indices
 
* logging/verbosity control
   * boost.log 
      * unfortunately not yet in distros
      * was provisionally approved but v1 looked difficult to use, TODO: check v2

* command line parsing when have duplicated options (like -o) 
   gives "multiple occurrences", change handling to
       * last one wins
       * OR immediate exit if it makes no sense for the option

* re-arrange python extension build to avoid littering wc with swig artifacts



Configurable loading of indices and generic access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``std::map <string,string>`` simple starting point implemented in r3436

Enable generic app indices by configuring queries providing (key,val) lists
which are loaded into ``std::map<string,XmlValue>``:: 

        [map.name]
        name = code2latex
        [map.query]	 
        query = for $glyph in collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='pdgs.xml' or dbxml:metadata('dbxml:name')='extras.xml' ]//glyph return (data($glyph/@code), data($glyph/@latex)) 

Such maps could be accessible by generic extension function my:map('code2latex',$key )

Keeping qxml generic
~~~~~~~~~~~~~~~~~~~~~

dlopen/dlsym (or C++ equivalent) handling for resolver and 
extension functions to prevent project specifics from creeping into qxml.
Such specifics should be being developed elsewhere (in heprez repository for example).

Some generic extfun will be needed however, so probably best to have an umbrella
resolver that handles
   * dynamic resolver loading
   * hands out resolve requests based on namespace uri.

See ~/env/dlfcn for tutorial of dlopen technique, the proxy registration 
approach described could be used to register per-library namespace keyed resolvers
which the umbrella resolver which lives in global main manages in a map.
 
http://www.faqs.org/docs/Linux-mini/C++-dlopen.html


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
    or dbxml-py 


Using C++ API
==============

.. warning:: DB XML docs have mismatches to header signatures, **trust headers above documentation**

Get there quick with dbxml-cpp

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

#. ``document-uri(root($smth))``  fails to provide the originating uri in more involved querying 
     * suspect a steps removed effect (fragments of fragments loose touch with their roots)
     * dbxml:metadata("dbxml:name",$smth)  seems to work OK without need to ``root`` up to the doc.

#. does not auto-coerce xs:string into xs:integer




