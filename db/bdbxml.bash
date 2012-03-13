# === func-gen- : db/bdbxml fgp db/bdbxml.bash fgn bdbxml fgh db
bdbxml-src(){      echo db/bdbxml.bash ; }
bdbxml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bdbxml-src)} ; }
bdbxml-vi(){       vi $(bdbxml-source) ; }
bdbxml-env(){      
   elocal- ; 
   export PATH=$PATH:$(bdbxml-bin)
   export BDBXML_HOME=$(bdbxml-home)
}
bdbxml-usage(){
  cat << EOU



installation
~~~~~~~~~~~~~

Had to sign up for Oracle web account before manually downloading::

    mv  ~/Downloads/dbxml-2.5.16.tar .

For installation docs::

    open file:///usr/local/env/db/dbxml-2.5.16/dbxml/docs/ref_xml/xml_unix/intro.html

Try fully default build (took ~2hrs)::

    sh buildall.sh

 /usr/local/env/db/dbxml-2.5.16/install/lib



python bindings installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Module ships with python::

	g4pb:~ blyth$ python -c "import bsddb ; print bsddb.__version__ "
	4.4.5.3


From /usr/local/env/db/dbxml-2.5.16/dbxml/src/python/README version match requirements
that are likely not to be met.


BUILDING bsddb3 module
----------------------

* remember dbxml is grafted ontop of a bdb installation, so nefarious scripts abound 


First build and install against naked macports py::

	g4pb:~ blyth$ python -V
	Python 2.5.5
	g4pb:~ blyth$ which python
	/opt/local/bin/python

	cd /usr/local/env/db/dbxml-2.5.16/dbxml/src/python/bsddb3-4.8.1
	python setup.dbxml.py build 
	...
	running build_ext
	building 'bsddb3._pybsddb' extension
	creating build/temp.macosx-10.5-ppc-2.5
	creating build/temp.macosx-10.5-ppc-2.5/Modules
	/usr/bin/gcc-4.0 -fno-strict-aliasing -mno-fused-madd -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -DPYBSDDB_STANDALONE=1 -I../../../../install/include -I/opt/local/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 -c Modules/_bsddb.c -o build/temp.macosx-10.5-ppc-2.5/Modules/_bsddb.o
	/usr/bin/gcc-4.0 -L/opt/local/lib -arch ppc -bundle -undefined dynamic_lookup build/temp.macosx-10.5-ppc-2.5/Modules/_bsddb.o -L../../../../install/lib -L../../../../install/lib -ldb -o build/lib.macosx-10.5-ppc-2.5/bsddb3/_pybsddb.so

	
	sudo python setup.dbxml.py install
	...
	creating dist
	creating 'dist/bsddb3-4.8.1-py2.5-macosx-10.5-ppc.egg' and adding 'build/bdist.macosx-10.5-ppc/egg' to it
	removing 'build/bdist.macosx-10.5-ppc/egg' (and everything under it)
	Processing bsddb3-4.8.1-py2.5-macosx-10.5-ppc.egg
	creating /opt/local/lib/python2.5/site-packages/bsddb3-4.8.1-py2.5-macosx-10.5-ppc.egg
	Extracting bsddb3-4.8.1-py2.5-macosx-10.5-ppc.egg to /opt/local/lib/python2.5/site-packages
	Adding bsddb3 4.8.1 to easy-install.pth file

	Installed /opt/local/lib/python2.5/site-packages/bsddb3-4.8.1-py2.5-macosx-10.5-ppc.egg
	Processing dependencies for bsddb3==4.8.1
	Finished processing dependencies for bsddb3==4.8.1


Quick check::

	g4pb:~ blyth$ python -c "import bsddb3 as _ ; print _.__file__,_.__version__ "
	/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/bsddb3-4.8.1-py2.5-macosx-10.5-ppc.egg/bsddb3/__init__.pyc 4.8.1


BUILDING the dbxml module
-------------------------

Following README to the letter, specifying ``--with-bsddb=bsddb3-4.8.1`` as just built that source::

	cd /usr/local/env/db/dbxml-2.5.16/dbxml/src/python
	python setup.py --with-bsddb=bsddb3-4.8.1 build
        ...
	building '_dbxml' extension
	creating build/temp.macosx-10.5-ppc-2.5
	/usr/bin/gcc-4.0 -fno-strict-aliasing -mno-fused-madd -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -DHAVE_BSDDB=1 -I/usr/local/env/db/dbxml-2.5.16/dbxml/src/python/bsddb3-4.8.1 -I/usr/local/env/db/dbxml-2.5.16/install/include -I/usr/local/env/db/dbxml-2.5.16/install/include -I/opt/local/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 -c dbxml_python_wrap.cpp -o build/temp.macosx-10.5-ppc-2.5/dbxml_python_wrap.o
	cc1plus: warning: command line option "-Wstrict-prototypes" is valid for C/ObjC but not for C++
	/usr/bin/g++-4.0 -L/opt/local/lib -arch ppc -bundle -undefined dynamic_lookup build/temp.macosx-10.5-ppc-2.5/dbxml_python_wrap.o -L/usr/local/env/db/dbxml-2.5.16/install/build_unix/.libs -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/build_unix/.libs -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/lib -L/usr/local/env/db/dbxml-2.5.16/install/lib -ldbxml -ldb-4 -lxqilla -lxerces-c -o build/lib.macosx-10.5-ppc-2.5/_dbxml.so

	sudo python setup.py install
	...
	copying build/lib.macosx-10.5-ppc-2.5/dbxml.py -> /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages
	byte-compiling /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/dbxml.py to dbxml.pyc
	running install_egg_info
	Writing /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/dbxml-2.5.16-py2.5.egg-info



TESTING and examples
--------------------

    cd ~/env/db/bdbxml/extpy
    ./test_versions.py

    cd $(dirname $BDBXML_HOME)/dbxml/examples/python
    python examples.py test
    python examples.py 1
    ...
    python examples.py 13


python functionality from XQuery/C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The py binding provides a complete python interface to the C++ API, 
but there seems no way to use py functionality from C++. 
Cannot register a python resolver with a C++ manager ?

    examples/python/misc/externalFunction.py

This is a bit different from exist extension where (for example)
XQuery functions call out to python (actually jython).

However, as are all taking to the same container, can write new docs
as a result of each languages strengths.  This way is actually more
direct, and easier to develop/debug than the Russian dolls approach.



Documentation
~~~~~~~~~~~~~~

    file:///usr/local/env/db/dbxml-2.5.16/dbxml/docs/index.html
    file:///usr/local/env/db/dbxml-2.5.16/dbxml/docs/intro_xml/index.html    
    file:///usr/local/env/db/dbxml-2.5.16/dbxml/docs/gsg_xml/cxx/index.html


dbxml command line tool
~~~~~~~~~~~~~~~~~~~~~~~

#. no readline support, FAQ suggests http://freecode.com/projects/rlwrap
#. how to determine document names 
#. tis tedious having to quote queries

adding indices
~~~~~~~~~~~~~~~


dbxml> time query '
collection("parts.dbxml")/part[parent-part]'
10000 objects returned for eager expression '
collection("parts.dbxml")/part[parent-part]'

Time in seconds for command 'query': 6.43564


dbxml> addIndex "" parent-part node-element-presence-none
Adding index type: node-element-presence-none to node: {}:parent-part

   ## this took minutes !!!

dbxml> time query '
collection("parts.dbxml")/part[parent-part]'
10000 objects returned for eager expression '
collection("parts.dbxml")/part[parent-part]'

Time in seconds for command 'query': 0.565597

     ## more than factor 10 improvement 

dbxml> time query '
collection("parts.dbxml")/part[parent-part = 1]'
3333 objects returned for eager expression '
collection("parts.dbxml")/part[parent-part = 1]'

Time in seconds for command 'query': 0.881593



dbxml> addIndex "" parent-part node-element-equality-double
Adding index type: node-element-equality-double to node: {}:parent-part

   ## again several minutes to add the index

dbxml> time query '
collection("parts.dbxml")/part[parent-part = 1]'
3333 objects returned for eager expression '
collection("parts.dbxml")/part[parent-part = 1]'

Time in seconds for command 'query': 0.256801         ## factor 3-4


node-element-string-equality
node-attribute-string-equality
node-element-double-equality
node-attribute-double-equality 

schema
~~~~~~~~

  file:///usr/local/env/db/dbxml-2.5.16/dbxml/docs/intro_xml/schema.html

#. can schema location be global to a container, or does it have to be specified in the instance docs ?


load eXist backup into Berkeley DB XML container 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using db/bdbxml/migrate.cc create a 7.3MB dbxml container with::

   ./migrate /data/heprez/data/backup/part/localhost/2012/Mar06-1922/db/hfagc /tmp/hfagc.dbxml


query interactively with dbxml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

	dbxml> openContainer  /tmp/hfagc.dbxml   ## just content of /db/hfagc

	dbxml> addAlias hfagc            # for the default container, that can be used in collection functions

	dbxml> setNamespace rez http://hfag.phys.ntu.edu.tw/hfagc/rez
	Binding rez -> http://hfag.phys.ntu.edu.tw/hfagc/rez

	dbxml> query "collection('hfagc')//rez:rez"
	226 objects returned for eager expression 'collection('hfagc')//rez:rez'

	dbxml> query "collection('hfagc')//rez:quote"
	731 objects returned for eager expression 'collection('hfagc')//rez:quote'

	dbxml> time query "collection('hfagc')//rez:quote[rez:qtag='BR:-511:225,443']"
	3 objects returned for eager expression 'collection('hfagc')//rez:quote[rez:qtag='BR:-511:225,443']'

	Time in seconds for command 'query': 0.007367

	dbxml> q collection()/rez:rez                                        # default container is used
	226 objects returned for eager expression 'collection()/rez:rez'

	dbxml> time q 'for $a in collection() return dbxml:metadata("dbxml:name", $a)'
	226 objects returned for eager expression 'for $a in collection() return dbxml:metadata("dbxml:name", $a)'

	Time in seconds for command 'q': 0.018299

	dbxml> time q collection()/*[dbxml:metadata('dbxml:name')='cdf/cjl/cdf_summer2007_BsDsK.xml']
	1 objects returned for eager expression 'collection()/*[dbxml:metadata('dbxml:name')='cdf/cjl/cdf_summer2007_BsDsK.xml']'

	Time in seconds for command 'q': 0.006099



integration with codesynthesis XSD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See /usr/local/env/xml/xsd-3.3.0-powerpc-macosx/examples/cxx/tree/dbxml/driver.cxx for 
example of creating an object model from document fragment results of BDB XML XQuery.
Can also add custom methods to the model and also hookup functions external to XQuery implemented in the C++.



EOU
}
bdbxml-dir(){ echo $(local-base)/env/db/$(bdbxml-name) ; }
bdbxml-bin(){ echo $(bdbxml-dir)/install/bin ; }
bdbxml-cd(){  cd $(bdbxml-dir); }
bdbxml-mate(){ mate $(bdbxml-dir) ; }
bdbxml-home(){ echo $(bdbxml-dir)/install ; }
bdbxml-url(){ echo http://download.oracle.com/otn/berkeley-db/$(bdbxml-name).tar.gz ; }
bdbxml-name(){ echo dbxml-2.5.16 ; }
bdbxml-get(){
   local dir=$(dirname $(bdbxml-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(bdbxml-url)
   local bas=$(basename $url)
   local nam=$(bdbxml-name)
   # [ ! -f "$bas" ] && curl -L -O $url

   [ ! -d "$nam" -a -f "$nam.tar" ] && tar xvf $nam.tar   

}

bdbxml-test(){
  local tmp=/tmp/env/db/bdbxml
  mkdir -p $tmp
  cd $tmp

  local bkp=/data/heprez/data/backup/part/localhost/2012/Mar06-1922

}


bdbxml-swig-notes(){ cat << EON

Notice optimizing flags:: 

    swig -python -help


g4pb:swig blyth$ pwd
/usr/local/env/db/dbxml-2.5.16/dbxml/dist/swig

g4pb:swig blyth$ grep XmlExternalFunction *.i
dbxml.i:class XmlExternalFunction;
dbxml.i:        virtual XmlExternalFunction *resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
dbxml.i:class XmlExternalFunction
dbxml.i:        XmlExternalFunction() {}
dbxml.i:        virtual ~XmlExternalFunction() {}
dbxml_python.i:%feature("director") XmlExternalFunction;
dbxml_python.i:%exception XmlExternalFunction::execute {
dbxml_python.i:%exception XmlExternalFunction::close {


/usr/local/env/db/dbxml-2.5.16/dbxml/dist/swig/dbxml.i:1956: Warning(473): Returning a pointer or reference in a director method is not recommended.
/usr/local/env/db/dbxml-2.5.16/dbxml/dist/swig/dbxml.i:1959: Warning(473): Returning a pointer or reference in a director method is not recommended.
/usr/local/env/db/dbxml-2.5.16/dbxml/dist/swig/dbxml.i:1965: Warning(473): Returning a pointer or reference in a director method is not recommended.
/usr/local/env/db/dbxml-2.5.16/dbxml/dist/swig/dbxml.i:1968: Warning(473): Returning a pointer or reference in a director method is not recommended.



g4pb:qxml blyth$ python -c "import pyextfun"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    File "pyextfun.py", line 7, in <module>
        import _pyextfun
	ImportError: dlopen(./_pyextfun.so, 2): Symbol not found: __ZN5DbXml10XmlResultsD1Ev
	  Referenced from: /Users/blyth/env/db/bdbxml/qxml/_pyextfun.so
	    Expected in: dynamic lookup




EON
}

bdbxml-swig(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local tmp=/tmp/env/$FUNCNAME
   mkdir -p $tmp && cd $tmp
	      
   # extracts from /usr/local/env/db/dbxml-2.5.16/dbxml/dist/s_swig  
   local swigd=$(bdbxml-dir)/dbxml/dist/swig

   local swigi=$swigd/dbxml.i
   local swigl=$(basename $swigi)
   local outd=python
   local wrap=$outd/dbxml_python_wrap.cpp

   mkdir -p $outd


   if [ -d "$outd" -a -f "$wrap"   ]; then 
       echo $msg already generated wrap $wrap
   else
      #	   
      # all these are %include	from dbxml.i but cannot find most of them ?   
      # dbxml.i exception.i typemaps.i std_string.i dbxml_python.i"
      #
      cp $swigd/dbxml.i .
      # cp $swigd/dbxml_python.i .

      swig -Wall -python -c++ -threads -I$swigd -outdir $outd -o $wrap $swigl
      sed -f $swigd/python-post.sed $wrap > $wrap.tmp
      cp $wrap.tmp $wrap
   fi   

   local actd=$(bdbxml-dir)/dbxml/src/python

   # lots of differences : maybe swig version effect 
   #diff $actd/dbxml_python_wrap.cpp $outd/dbxml_python_wrap.cpp  
   #diff $actd/dbxml_python_wrap.h   $outd/dbxml_python_wrap.h  
   #diff $actd/dbxml.py              $outd/dbxml.py  

   #cd $iwd
}



