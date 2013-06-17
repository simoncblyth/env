
QXML Examples
===============

.. contents:: :local:


STDIN query piping and shebang line running
---------------------------------------------

Usage examples assuming bash shell.
Piping from echo (must escape some chars from shell, handy for one-liners)::

     echo collection\(\)[1]               | qxml -
     echo "collection()[1]//rez:quote[1]" | qxml -   # first quote of first item
     echo "count(collection())"           | qxml -   # number of items
     echo "count(collection()/rez:rez)"   | qxml -   # 

NB use of configured default container avoids::

    echo "collection('dbxml:/hfc')[1]"   | qxml -
    echo "collection('dbxml:/tmp')[1]"   | qxml -
          ## containers identified by configured aliases or tags

Three queries that are the same::

    echo "collection('dbxml:////tmp/hfagc/hfagc.dbxml')/dbxml:metadata('dbxml:name')" | qxml -  ## explicit file path to the container
    echo "collection('hfc')/dbxml:metadata('dbxml:name')" | qxml -     # using the tag name  
    echo "collection()/dbxml:metadata('dbxml:name')" | qxml -          # using the default container

Useful for quick syntax checking::

     echo "let \$a := (1,2,3,4,5) return \$a[last()] " | qxml -
     echo 'let  $a := (1,2,3,4,5) return  $a[last()] ' | qxml -
     echo 'let $a := (1,2,3,4,5) return $a[position() <= 3] ' | qxml -

Grabbing a resource by name::

     echo "collection()/*[dbxml:metadata('dbxml:name')='/cdf/cjl/cdf_summer2007_BsDsK.xml']" | qxml -
     echo "collection()/*[dbxml:metadata('dbxml:name')='/cdf/cjl/cdf_summer2007_BsDsK.xml']" | qxml - > out.xml
          # redirect stdout to file, 
          # meta output goes to stderr allowing queries to yield valid XML

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



Mapping element nodes in larger docs, eg SVG
----------------------------------------------

Element handle do not encode the container and will become invalid if document changed. Usable from XQuery::

    echo 'let $nod := doc("tmp/qtag2latex.xml")//qtag[10] let $hdl := $nod/dbxml:node-to-handle() return ($nod,$hdl,dbxml:handle-to-node("tmp",$hdl))' | qxml -

And in C++::

    string hdl = val.getNodeHandle(); // from XmlValue ... 
    cont.getNode(hdl);


Issues/Enhancements/Ideas
---------------------------

* install more python into $ENV_PREFIX/lib/ to allow use from anywhere 

* ensure qxml return codes are appropriate when xquery or other errors occur

* make configured maps to load command line controllable, check error handling when maps are missing 

* avoid absolute paths in config file 
   * maybe allow envvar interpolation for a list of named envvars, eg HEPREZ_HOME QXML_TMP 
      * use python style eg  %(HEPREZ_HOME)s/some/path  
           * makes python implementation easy  
           * cpp easier than shell style $HEPREZ_HOME/some/path  
   *  documentation mentions that container resolution defaults be being relative to the environment dir 
      currently have not used this, instead have been specifying absolute paths in config and using them via aliases.
      Potentially moving to relative addressing could cut down the number of absolute paths in the config.       

* ``dbxml`` shell like capabilities for ``qxml`` that hook into the qxml configuration  
   /usr/local/env/db/dbxml-2.5.16/dbxml/src/utils/shell/dbxmlsh.cpp

* resolver rationalization : THIS IS TIED TO DYLIB LOADING 
   * python resolver / C++ resolver / swigged C++ resolver 
   * python resolver palm off to swigged C++ resolver ? 
   * separate ns for python and swigged C++ for implementation comparisons

* configuration of dbxml indices
 
* logging/verbosity control
   * boost.log 
      * unfortunately not yet in distros
      * was provisionally approved but v1 looked difficult to use, TODO: check v2

* command line parsing when have duplicated options (like -o) 
   gives "multiple occurrences", change handling to
       * last one wins
       * OR immediate exit if it makes no sense for the option

* re-arrange python extension build to avoid littering wc with swig artifacts

Done
~~~~~~

 * on writing xml into dbxml containers fill in created/modified/owner metadata


Configurable loading of indices and generic access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``std::map <string,string>`` simple starting point implemented in r3436

Enable generic app indices by configuring queries providing (key,val) lists
which are loaded into ``std::map<string,XmlValue>``:: 

        [map.name]
        name = code2latex
        [map.query]	 
        query = for $glyph in collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='pdgs.xml' or dbxml:metadata('dbxml:name')='extras.xml' ]//glyph return (data($glyph/@code), data($glyph/@latex)) 

Such maps are accessible by generic extension function my:map('code2latex',$key )

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
 
* http://www.faqs.org/docs/Linux-mini/C++-dlopen.html


Steps to add a C++ extension function
----------------------------------------

#. implement in ``extfun.{cc,hh}`` 
#. add argument signature to ``extresolve.cc``
#. build C++ qxml with ``make``
#. add test calls to ``test/ext.xq`` that exercise the extension


Steps to make C++ extension available from python main XQueries
----------------------------------------------------------------

#. setup swig wrapping in ``extfun.i``

Using the python API
----------------------

Very close to C++, but not the same need to examine::

    vi /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/dbxml.py
    or dbxml-py 

Using C++ API
---------------

.. warning:: DB XML docs have mismatches to header signatures, **trust headers above documentation**

Get there quick with dbxml-cpp

Using dbxml shell
------------------

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



dbxml shell as debugging tool
--------------------------------

It can be very useful to use the dbxml shell for debugging without all the conveniences of qxml
getting in the way. For example whilst debugging a single resource transfer script found that 
it seemed to transfer OK but created invisible docs from the point of view of qxml queries like::

    echo 'for $a in collection("sys") return $a/dbxml:metadata("dbxml:name")' | qxml -
    echo 'for $a in collection("sys") return $a/dbxml:metadata("exist:name")' | qxml -


environment
------------

::

    simon:dayabay blyth$ dbxml -h /tmp/dbxml
    -bash: dbxml: command not found
    simon:dayabay blyth$ db-
    simon:dayabay blyth$ bdbxml-
    simon:dayabay blyth$ dbxml -h /tmp/dbxml
    Joined existing environment

    dbxml> 

Examples
---------

Checking with dbxml shell (**have observed inconsistencies when not using the same envdir as qxml**)::

   dbxml -h /tmp/dbxml
   Joined existing environment

   dbxml> openContainer /tmp/hfagc/hfagc_system.dbxml
   dbxml> query 'for $a in collection() return $a/dbxml:metadata("dbxml:name")'
   16 objects returned for eager expression 'for $a in collection() return $a/dbxml:metadata("dbxml:name")'

   dbxml> print
   pdgs.xml
   pdg.xml
   ...

   dbxml> getDocument lhcb_winter2011_BcX.xml
   1 documents found

   dbxml> print 
   <rez:rez xmlns:rez="http://hfag.phys.ntu.edu.tw/hfagc/rez" xmlns:exist="http://exist.sourceforge.net/NS/exist" exist:id="1" exist:source="/db/test/lhcb_winter2011_BcX.xml">
	    <rez:header mode="pro" time="2012-04-08T00:36:44.088+0800" stamp="1333816604088" stamp_hash="ixml:content-hash:lhcb_winter2011_BcX.xml/db/hfagc/lhcb/yasmine/lhcb_winter2011_BcX.xmlyasminelhcb2012-04-08T00:21:26.978+08:001.0-dev/data/heprez/install/exist/eXist-snapshot-20051026/unpack/4" stamp_id="2" stamp_source="/db/hfagc/lhcb/yasmine/lhcb_winter2011_BcX.xml">
		<rez:origin>
                ...


Hmm, hfagc system container empty
----------------------------------- 

::

    simon:dayabay blyth$ dbxml -h /tmp/dbxml
    Joined existing environment

    dbxml>  openContainer /tmp/hfagc/hfagc_system.dbxml

    dbxml> query 'for $a in collection() return $a/dbxml:metadata("dbxml:name")'
    0 objects returned for eager expression 'for $a in collection() return $a/dbxml:metadata("dbxml:name")'

    dbxml> print

    dbxml> openContainer /tmp/hfagc/hfagc.dbxml

    dbxml>  query 'for $a in collection() return $a/dbxml:metadata("dbxml:name")'
    256 objects returned for eager expression 'for $a in collection() return $a/dbxml:metadata("dbxml:name")'

    dbxml> print
    /babar/cecilia/b0d0kpi.xml
    /babar/cecilia/b0dsa02.xml
    /babar/cecilia/b0dspi.xml
    /babar/cecilia/b0dsstardstar.xml
    ...




Observations on Berkeley DB XML XQuerying
-------------------------------------------

#. ``document-uri(root($smth))``  fails to provide the originating uri in more involved querying 
     * suspect a steps removed effect (fragments of fragments loose touch with their roots)
     * dbxml:metadata("dbxml:name",$smth)  seems to work OK without need to ``root`` up to the doc.

#. does not auto-coerce xs:string into xs:integer

#. using baseuri setting in hfagc.ini ``dbxml.baseuri = dbxml:/`` (this is the default without qxml) affords collection specification by alias alone::

      echo 'for $a in tokenize("tmp hfc sys"," ") return count(collection($a)) ' | qxml -     
      echo 'doc("tmp/qtag2latex.xml")' | qxml -
      echo 'for $q in doc("tmp/qtag2latex.xml")//qtag return (data($q/@value),data($q/latex))' | qxml -
      echo 'for $q in doc("tmp/qtag2latex.xml")//qtag return ($q/@value/string(),$q/latex/string())' | qxml -     
 
      ## CAUTION WITH SHELL ESCAPING 

