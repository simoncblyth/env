# === func-gen- : xml/xsd fgp xml/xsd.bash fgn xsd fgh xml
xsd-src(){      echo xml/xsd.bash ; }
xsd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(xsd-src)} ; }
xsd-vi(){       vi $(xsd-source) ; }
xsd-env(){      elocal- ; }
xsd-usage(){
  cat << EOU


XSD : compile XSD schema into C++ model classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://www.codesynthesis.com/products/xsd/download.xhtml

file:///usr/local/env/xml/xsd-3.3.0-powerpc-macosx/documentation/xsd.xhtml
file:///usr/local/env/xml/xsd-3.3.0-powerpc-macosx/documentation/schema-authoring-guide.xhtml
file:///usr/local/env/xml/xsd-3.3.0-powerpc-macosx/documentation/cxx/tree/guide/index.xhtml


Preqs
~~~~~~~

sudo port install xercesc

port info xercesc
xercesc @2.8.0, Revision 2 (textproc)

port info xercesc3
xercesc3 @3.1.1, Revision 4 (textproc, xml, shibboleth)


g4pb:scrape blyth$ sudo port install xercesc
Password:
--->  Fetching archive for xercesc
--->  Attempting to fetch xercesc-2.8.0_2.darwin_9.ppc.tgz from http://packages.macports.org/xercesc
--->  Fetching xercesc
--->  Attempting to fetch xerces-c-src_2_8_0.tar.gz from http://mirror.internode.on.net/pub/apache/xerces/c/2/sources/
--->  Verifying checksum(s) for xercesc
--->  Extracting xercesc
--->  Applying patches to xercesc
--->  Configuring xercesc
--->  Building xercesc
--->  Staging xercesc into destroot
--->  Installing xercesc @2.8.0_2
--->  Activating xercesc @2.8.0_2
--->  Cleaning xercesc



hello
~~~~~~

   cd examples/cxx/tree/hello
   make CXXFLAGS="-I/opt/local/include" LDFLAGS="-L/opt/local/lib"    ## use macports Xerces-c


feed some real schema
~~~~~~~~~~~~~~~~~~~~~~

    xsd-







EOU
}
xsd-name(){ echo xsd-3.3.0-powerpc-macosx ; }
xsd-dir(){ echo $(local-base)/env/xml/$(xsd-name) ; }
xsd-cd(){  cd $(xsd-dir); }

xsd-url(){ echo http://www.codesynthesis.com/download/xsd/3.3/macosx/powerpc/$(xsd-name).tar.bz2 ; }
xsd-mate(){ mate $(xsd-dir) ; }
xsd-get(){
   local dir=$(dirname $(xsd-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(xsd-url)
   local bz2=$(basename $url)
   local tar=${bz2/.bz2}
   local nam=${tar/.tar}

   [ -d "$nam" ] && return 0
   [ -f "$tar" ] && tar xvf $tar  && return
   [ -f "$bz2" ] && bunzip2 $bz2  && return
   [ ! -f "$nam" ] && curl -L -O $url

}


xsd-bin(){ echo $(xsd-dir)/bin/xsd ; }
xsd(){ $(xsd-bin) $* ; }

xsd-rez(){

   local tmp=/tmp/env/$FUNCNAME
   mkdir -p $tmp
   cd $tmp

   local schema=~/heprez/chiba/schema-cradle/rez-v003.xsd 
   local name=$(basename $schema)

   cp $schema $name
   xsd cxx-tree --show-sloc $name


}


