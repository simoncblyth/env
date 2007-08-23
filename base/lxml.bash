
##   http://codespeak.net/lxml/installation.html
lxml-install(){
   easy_install lxml

  #
  # on hfag this fails ... because of non-standard libxml2 and libxslt locations ...
  # on OS X the framework magic prevents this (?)
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


lxml-get(){

   local dir=$LOCAL_BASE/python/lxml
   mkdir -p $dir
   
   local name=lxml-1.3.3
   local tgz=$name.tgz
   local url=http://codespeak.net/lxml/$tgz
   
   test -f $tgz || curl -o $tgz $url
   test -d $name || tar zxvf $tgz 

}



