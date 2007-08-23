
libxml2-env(){

  local vers=2.6.29
  export LIBXML2_VERSION=$vers
  export LIBXML2_NAME=libxml2-$vers
  export LIBXML2_FOLD=$LOCAL_BASE/libxml2

}


libxml2-get(){

    libxml2-env

    local dir=$LOCAL_BASE/libxml2
    [ -d $dir ] || ( sudo mkdir -p $dir && sudo chown $USER $dir ) 
   
    local name=$LIBXML2_NAME
    local tgz=$name.tar.gz
    local url=ftp://xmlsoft.org/libxml2/$tgz
    
    cd $dir
    test -f $tgz || curl -o $tgz $url
    test -d $name || tar zxvf $tgz 
}


libxml2-dir(){

    libxml2-env
    local dir=$LOCAL_BASE/libxml2/$LIBXML2_NAME
    test -d $dir || ( echo error no folder $dir && return 1 ) 
    cd $dir
}

libxml2-configure(){

    libxml2-dir
    ./configure --prefix=$LIBXML2_FOLD --with-python=$PYTHON_HOME

#
# checking for python... /data/usr/local/python/Python-2.5.1/bin/python
# Found Python version 2.5
# could not find python2.5/Python.h
#
#
# need to have the  --with-python=$PYTHON_HOME to find the python headers...
#
# Found python in /data/usr/local/python/Python-2.5.1/bin/python
# Found Python version 2.5
#

}




libxml2-make(){
  
    libxml2-dir 
    make

# make[3]: Leaving directory `/data/usr/local/libxml2/libxml2-2.6.29/doc'
# I/O error : Attempt to load network entity http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd
# ../doc/news.html:2: warning: failed to load external entity "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"
#  1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"
#  
#                                                                               
#  make[3]: Entering directory `/data/usr/local/libxml2/libxml2-2.6.29/python'
# /data/usr/local/python/Python-2.5.1/bin/python ./generator.py .
# /data/usr/local/python/Python-2.5.1/lib/python2.5/xmllib.py:9: DeprecationWarning: The xmllib module is obsolete.  Use xml.sax instead.
#  warnings.warn("The xmllib module is obsolete.  Use xml.sax instead.", DeprecationWarning)
# Found 1595 functions in libxml2-api.xml
# Found 55 functions in libxml2-python-api.xml
# Generated 880 wrapper functions, 555 failed, 215 skipped
#
# Missing type converters: 
# xmlRelaxNGValidityErrorFunc *:2  xmlXPathObjectPtr:22  const htmlEntityDesc *:2  xmlOutputMatchCallback:1  xmlElementContentPtr *:1  xmlStructuredErrorFunc:5  xmlSchematronValidCtxtPtr:2  xmlParserInputBufferCreateFilenameFunc:2  xmlSchemaValType:2  size_t:1  xmlEnumerationPtr:5  xmlSchemaWildcardPtr:1  xmlXIncludeCtxtPtr:4  xmlRelaxNGValidityErrorFunc:2  xmlSAXHandler *:4  ...:1  xmlShellReadlineFunc:1  xmlDict *:1  xmlAutomataPtr:19  xmlParserInputPtr:17  xmlCatalogAllow:2  xmlExpNodePtr:3  xmlElementContent *:1  xmlCharEncodingOutputFunc:1  xmlDictPtr:10  xmlTextWriterPtr:77  const htmlElemDesc *:1  xmlChRangeGroup *:1  xmlIDPtr:1  xmlSchemaValPtr:13  xmlInputMatchCallback:1  xmlElementTablePtr:2  xmlChar **:16  xmlXPathCompExprPtr:6  xmlTextReaderErrorFunc:1  xmlExternalEntityLoader:2  xmlNotationTablePtr:2  xmlXPathVariableLookupFunc:1  xmlParserNodeInfoPtr:1  xmlExpCtxtPtr:14  xmlPatternPtr:8  xmlC14NIsVisibleCallback:1  xmlDeregisterNodeFunc:2  va_list:1  xmlSchemaTypePtr:9  htmlStatus:1  xmlRegisterNodeFunc:2  xmlAttributeType:2  xmlRefPtr:1  xmlCharEncodingHandler *:4  xmlNotationPtr:3  xmlSaveCtxtPtr:8  xmlRegExecCallbacks:1  xmlNsPtr *:1  xmlLocationSetPtr:6  xmlSchemaSAXPlugPtr:1  xmlModulePtr:4  xmlEnumerationPtr *:2  xmlShellCtxtPtr:10  xlinkNodeDetectFunc:2  xmlRefTablePtr:1  xmlStreamCtxtPtr:6  xmlSchemaValidityErrorFunc *:2  xmlAttributeTablePtr:2  xmlSchematronParserCtxtPtr:5  xmlCatalogPrefer:1  xmlParserNodeInfoSeqPtr:3  xmlSchematronPtr:2  xmlNodePtr *:2  xmlInputReadCallback:7  char **:5  xmlCharEncoding:13  xmlRegExecCtxtPtr:5  xmlElementContentType:2  void *:86  xmlTextReaderErrorFunc *:1  xmlSAXHandlerPtr *:1  xmlEntityReferenceFunc:1  xmlDocPtr *:1  xmlBufferAllocationScheme:3  xmlSchemaValidityErrorFunc:2  xmlDOMWrapCtxtPtr:6  xmlOutputWriteCallback:2  xmlSchemaFacetPtr:7  xlinkHandlerPtr:2  xmlXPathFuncLookupFunc:1  htmlElemDesc *:3  xmlCharEncodingHandlerPtr:7  xmlCharEncodingInputFunc:1  xmlFeature:1  const xmlParserNodeInfo *:1  xmlNodeSetPtr:32  xmlEntitiesTablePtr:3  xmlIDTablePtr:1  xmlXPathFunction:4  xmlOutputBufferCreateFilenameFunc:2  xmlElementContentPtr:8  xmlElementTypeVal:1  xlinkType:1  xmlGenericErrorFunc *:1                                                                                                                                                            
#                                                                                                                                                                                                                                                                                                                     ^

}





libxml2-install(){
  
    libxml2-dir 
    make install
}


libxml2-tests(){
    
    libxml2-dir
    make tests
    
    
    
}




libxml2-get-rpms(){

  ## not pursued ... rpms are linux specific so prefer not to follow this route 

    libxml2-env
     
    local vers=$LIBXML2_VERSION    
    local name=$LIBXML2_NAME
    local dame=libxml2-devel-$vers
    
    local srpm=$name-1.src.rpm
    local drpm=$dame-1.i386.rpm
    
    # there is no devel rpm for that version ... so try the tgz route 
    
    cd $dir

    local rpms="$srpm $drpm"
    for rpm in $rpms
    do
        
       local url=ftp://xmlsoft.org/libxml2/$rpm
       echo === libxml2-get $url to $rpm 
       test -f $rpm || curl -o $rpm $url
    done
    
  




}

