
tracxsltmacro-get(){

   # http://www.trac-hacks.org/wiki/XsltMacro

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins

   local macro=xsltmacro
   mkdir -p $macro
   svn co http://trac-hacks.org/svn/$macro/0.9/ $macro
 

}