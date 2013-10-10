
lxml-vi(){ vi $(env-home)/base/lxml.bash ; }
lxml-usage(){ cat << EOU

lxml
=====

::

    [blyth@belle7 ~]$ sudo yum --enablerepo=epel install python-lxml

EOU
}


##   http://codespeak.net/lxml/installation.html
lxml-install(){
   easy_install lxml

  #
  # on hfag this fails ... because of non-standard libxml2 and libxslt locations ...
  # on OS X the ~/Library/Frameworks magic prevents this (?)
  #   
  # fix the issue below in lxml-build by setting options to python setup.py build_ext ... 
  #

# [blyth@hfag blyth]$ easy_install lxml
# Searching for lxml
# Best match: lxml 1.3.3
# Downloading http://codespeak.net/lxml/lxml-1.3.3.tgz
# Processing lxml-1.3.3.tgz
# Running lxml-1.3.3/setup.py -q bdist_egg --dist-dir /tmp/easy_install--l6Vzc/lxml-1.3.3/egg-dist-tmp-MHKVz7
# Building lxml version 1.3.3
# warning: no previously-included files found matching 'doc/pyrex.txt'
# warning: no previously-included files found matching 'src/lxml/etree.pxi'
# In file included from src/lxml/etree.c:22:
# src/lxml/etree_defs.h:40:31: libxml/xmlversion.h: No such file or directory
# src/lxml/etree.c:27:31: libxml/xmlversion.h: No such file or directory
# src/lxml/etree.c:28:29: libxml/encoding.h: No such file or directory
# src/lxml/etree.c:29:28: libxml/chvalid.h: No such file or directory
# src/lxml/etree.c:30:25: libxml/hash.h: No such file or directory
# src/lxml/etree.c:31:25: libxml/tree.h: No such file or directory
# src/lxml/etree.c:32:26: libxml/valid.h: No such file or directory
# src/lxml/etree.c:33:26: libxml/xmlIO.h: No such file or directory
#...
#




}


lxml-env(){

   #export LXML_NAME=lxml-1.3.3
   export LXML_NAME=lxml-2.0.1
   export LXML_FOLD=$LOCAL_BASE/python/lxml
}



lxml-get(){

   lxml-env
   local dir=$LXML_FOLD
   $SUDO mkdir -p $dir  && $SUDO chown -R $USER $dir 
   
   local name=$LXML_NAME
   local tgz=$name.tgz
   local url=http://codespeak.net/lxml/$tgz
   
    cd $dir
   test -f $tgz || curl -o $tgz $url
   test -d $name || tar zxvf $tgz 

}

lxml-dir(){

   lxml-env
   local dir=$LXML_FOLD/$LXML_NAME
   test -d $dir || ( echo no such dir $dir && return 1 ) 
   cd $dir
}

lxml-build(){

   libxml2
   libxml2-env

   libxslt
   libxslt-env

##  from doc/build.txt :
##       python setup.py build_ext -i  -I $LIBXML2_FOLD
##    
##  looking at ez_setup.py seems that options are passed thru to setuptools...
##      easy_install -Z . -I $LIBXML2_FOLD
##      ... but doesnt work , says -I not recognized
##
  
    python setup.py build_ext  --help   ## hit the mother lode
    python setup.py build_ext --include-dirs=$LIBXML2_FOLD/include/libxml2:$LIBXSLT_FOLD/include --library-dirs=$LIBXML2_FOLD/lib:$LIBXSLT_FOLD/lib
    python setup.py install   


}

lxml-test(){

   python -c "import lxml"
}

