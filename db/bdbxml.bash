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


python bindings
~~~~~~~~~~~~~~~

http://jimmyg.org/blog/2008/oracle-db-xml-was-sleepycat.html

dbxml/src/python/README
dbxml/examples/python/examples.py
dbxml/examples/python/misc/externalFunction.py



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


