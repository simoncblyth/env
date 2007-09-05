
tractags-get(){

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins

   local url=http://trac-hacks.org/svn/tagsplugin/tags/0.4.1/
   local macro=tractags

   svn co $url  $macro  


}


