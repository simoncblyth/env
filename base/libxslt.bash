libxslt-env(){

  libxml2
  libxml2-env
  
  local vers
  if [ "$LIBXML2_VERSION" == "2.6.29" ]; then
     vers=1.1.21
  else
     echo no matched version 
     return 1 
  fi     
     
  export LIBXSLT_VERSION=$vers
  export LIBXSLT_NAME=libxslt-$vers
  export LIBXSLT_FOLD=$LOCAL_BASE/libxslt

}

libxslt-dir(){

    libxslt-env
    local dir=$LOCAL_BASE/libxslt/$LIBXSLT_NAME
    test -d $dir || ( echo error no folder $dir && return 1 ) 
    cd $dir
}



libxslt-get(){

    libxslt-env

    local dir=$LOCAL_BASE/libxslt
    [ -d $dir ] || ( sudo mkdir -p $dir && sudo chown $USER $dir ) 
   
    local name=$LIBXSLT_NAME
    local tgz=$name.tar.gz
    local url=ftp://xmlsoft.org/libxml2/$tgz
    
    cd $dir
    test -f $tgz || curl -o $tgz $url
    test -d $name || tar zxvf $tgz 
}



libxslt-configure(){

# http://jamesclarke.info/notes/libxml2

    libxslt-dir
	
   # ./configure \
   #  --with-python=$PYTHON_HOME \
   #  --prefix=$LIBXSLT_FOLD \
   #  --with-libxml-prefix=$LIBXML2_FOLD \
   #  --with-libxml-include-prefix=$LIBXML2_FOLD/include \
   #  --with-libxml-libs-prefix=$LIBXML2_FOLD/lib \
	 
#   on unix :  
#
# Found python in /data/usr/local/python/Python-2.5.1/bin/python
# PYTHON is pointing at /data/usr/local/python/Python-2.5.1/bin/python
# Found Python version 2.5
# Warning: Missing libxml2-python
#
#  after doing libxml2-py-pth ... can find libxml2
#
# Found python in /data/usr/local/python/Python-2.5.1/bin/python
# PYTHON is pointing at /data/usr/local/python/Python-2.5.1/bin/python
# Found Python version 2.5
# Found libxml2-python module
#
#	
#	
#	
#   for Leopard :	 
#	  	    
   ./configure \
     --prefix=$LIBXSLT_FOLD \
     --with-libxml-prefix=$LIBXML2_FOLD \
     --with-libxml-include-prefix=$LIBXML2_FOLD/include \
     --with-libxml-libs-prefix=$LIBXML2_FOLD/lib \
	 
	 


}


libxslt-make(){

  libxslt dir
  make $*
  
# make[3]: Entering directory `/data/usr/local/libxslt/libxslt-1.1.21/python'
# cd . && /data/usr/local/python/Python-2.5.1/bin/python generator.py
# /data/usr/local/python/Python-2.5.1/lib/python2.5/xmllib.py:9: DeprecationWarning: The xmllib module is obsolete.  Use xml.sax instead.
#   warnings.warn("The xmllib module is obsolete.  Use xml.sax instead.", DeprecationWarning)
# Found 235 functions in libxslt-api.xml
# Found 32 functions in libxslt-python-api.xml
# Generated 139 wrapper functions, 96 failed, 32 skipped
# 
# Missing type converters:
# xsltTopLevelFunction:2  xmlXPathObjectPtr:1  xsltDecimalFormatPtr:2  xmlChar **:2  xmlXPathCompExprPtr:4  xsltPreComputeFunction:1  xsltElemPreCompPtr:2  xsltDebugTraceCodes:2  xsltDocumentPtr:8  xsltSecurityPrefsPtr:11  xsltTemplatePtr:4  pythonObject *:5  ...:1  xsltNumberDataPtr:1  xmlHashTablePtr:1  xmlNodePtr *:3  xsltExtInitFunction:2  xsltCompilerCtxtPtr:2  char **:2  xmlXPathObjectPtr *:1  xmlNodeSetPtr:2  xmlXPathFunction:4  xsltTransformFunction:5  xsltCompMatchPtr:3  void *:13  xmlOutputBufferPtr:1  xsltPointerListPtr:4  xmlDictPtr:1  xsltSortFunc:2  xsltNsMapPtr:1  xsltStackElemPtr:3 
#  touch  
  
#
#   Leopard build looks similar to above tiger/unix one 
#
#  
  
  
}


#
#  libxslt-make tests > tests.log
#     mostly successful
#
#
# [g4pb:/usr/local/libxslt/libxslt-1.1.21] blyth$ libxslt-make tests > tests.log
# libtool: link: warning: `/usr/lib/gcc/powerpc-apple-darwin8/4.0.1/../../..//libiconv.la' seems to be moved
#



libxslt-install(){  libxslt-make install ; }

libxslt-py-pth(){
    libxslt-env
    echo $LIBXSLT_FOLD/lib/python2.5/site-packages  > $PYTHON_SITE/libxslt.pth
    
}

libxslt-py-test(){
   python -c "import libxslt"
}

