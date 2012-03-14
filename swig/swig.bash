# === func-gen- : swig/swig fgp swig/swig.bash fgn swig fgh swig
swig-src(){      echo swig/swig.bash ; }
swig-source(){   echo ${BASH_SOURCE:-$(env-home)/$(swig-src)} ; }
swig-vi(){       vi $(swig-source) ; }
swig-env(){      
   elocal- ; 
   #export SWIG_NAME=swig-1.3.29
   #export SWIG_HOME=$SYSTEM_BASE/swig/$SWIG_NAME
}
swig-usage(){
  cat << EOU



  http://www.swig.org/Doc2.0/SWIGDocumentation.html
  http://www.swig.org/Doc1.3/Python.html
  http://www.swig.org/Doc1.3/Customization.html#Customization_nn7


SWIG : wrap C/C++ to generate scripting language extensions
============================================================

see also ``swigbuild-``  

#.   http://www.swig.org/Doc2.0/SWIGDocumentation.html
#.   http://www.swig.org/Doc1.3/Python.html

Installs
~~~~~~~~

  ====================  ========  ===============================================================================================
  /usr/bin/swig          1.3.31    ancient system swig 
  ?                      1.3.39    created distributed dbxml_python_wrap.cpp in /usr/local/env/db/dbxml-2.5.16/dbxml/src/python 
  /opt/local/bin/swig    2.0.4     macports updated swig that resolved the Swig::DirectorTypeMismatchException head banging 
                                   in ~/env/db/bdbxml/qxml
  ====================  ========  ===============================================================================================



Basics
~~~~~~~

The bare %{ ... %} directive is a shortcut that is the same as %header %{ ... %}.
Other directives being: begin, runtime, header, wrapper, init
Everything in these is verbatim passed to the wrapper without being parsed by SWIG.



DB XML SWIG USAGE
~~~~~~~~~~~~~~~~~

/usr/local/env/db/dbxml-2.5.16/dbxml/dist/s_swig


OSX system swig
~~~~~~~~~~~~~~~~

g4pb:swig blyth$ which swig
/usr/bin/swig

swig -version

SWIG Version 1.3.31

Compiled with g++ [powerpc-apple-darwin9.0]
Please see http://www.swig.org for reporting bugs and further information


Simple Examples
~~~~~~~~~~~~~~~~

#. ``env/swig/hello/example.i``  python module from C functions
#. ``env/swig/hellocpp/Rectangle.i`` python module from C++ class
   

TODO : wrapping classes that use DB XML classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If can wrap a C++ class that consumes Berkeley DB XML XmlResult into a python 
module for example then can use that functionality from both XQuery C++ extensions 
and XQuery python extensions : avoiding language silos + implementation duplication 
at expense of learning how to swig wrap.

Thinking further, if could swig wrap a C++ ``XmlExternalFunction`` subclass such as
the below into a python module then can have the cake and eat it too. As this will allow:

#. import python wrapped XmlExternalFunction subclass into extfun.py and hookup to python resolver 
#. direct usage of C++ ``XmlExternalFunction`` subclass in extfun.cc with C++ resolver

::

	class MyExternalFunctionPow : public XmlExternalFunction
	{       
	   public:
		XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
		void close();

	};


This must be possible as the base is already wrapped::

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


Motivation for this is to allow flexible development whereby python prototyping 
can be ported as needed into C++ and that can be used during subsequent prototyping.




swig update
~~~~~~~~~~~~


g4pb:python blyth$ sudo port install swig
Password:
--->  Computing dependencies for bison
--->  Fetching archive for bison
--->  Attempting to fetch bison-2.5_0.darwin_9.ppc.tgz from http://packages.macports.org/bison
--->  Fetching bison
...
--->  Fetching swig
--->  Attempting to fetch swig-2.0.4.tar.gz from http://nchc.dl.sourceforge.net/project/swig/swig/swig-2.0.4
--->  Verifying checksum(s) for swig
--->  Extracting swig
--->  Configuring swig
--->  Building swig
--->  Staging swig into destroot
--->  Installing swig @2.0.4_0
--->  Activating swig @2.0.4_0
--->  Cleaning swig
g4pb:python blyth$ 



g4pb:python blyth$ sudo port install swig-python
Password:
--->  Computing dependencies for swig-python
--->  Fetching archive for swig-python
--->  Attempting to fetch swig-python-2.0.4_0.darwin_9.noarch.tgz from http://packages.macports.org/swig-python
--->  Fetching swig-python
--->  Verifying checksum(s) for swig-python
--->  Extracting swig-python
--->  Configuring swig-python
--->  Building swig-python
--->  Staging swig-python into destroot
--->  Installing swig-python @2.0.4_0
--->  Activating swig-python @2.0.4_0
--->  Cleaning swig-python
g4pb:python blyth$ 




EOU
}
swig-dir(){ echo $(local-base)/env/swig/swig-swig ; }
swig-cd(){  cd $(swig-dir); }
swig-mate(){ mate $(swig-dir) ; }
swig-get(){
   local dir=$(dirname $(swig-dir)) &&  mkdir -p $dir && cd $dir

}

