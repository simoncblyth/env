
tractags-get(){

   # http://www.trac-hacks.org/wiki/TagsPlugin

   cd $LOCAL_BASE/share/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins

   #local ver=0.4.1
   local ver=0.6 
   local url=http://trac-hacks.org/svn/tagsplugin/tags/$ver   
   local macro=tractags

   svn co $url  $macro  

}


tractags-install(){
      
   local macro=tractags
   cd $LOCAL_BASE/trac/plugins/$macro
   easy_install -Z .

   # Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracTags-0.4.1-py2.5.egg
}


tractags-unconf(){

   local name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini trac:default_handler:WikiModule

}

tractags-conf(){
  
   local msg="=== $FUNCNAME :"
   local name=${1:-$SCM_TRAC}
   echo $msg CAUTION changing the trac:default_handler:TagsWikiModule from WikiModule
   echo $msg this set up is for version 0.4 or 0.5 of tractags ... not the 0.6 version that is needed with 0.11 of trac 
   echo $msg see http://trac-hacks.org/wiki/TagsPlugin/0.5/Installation 
   
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini trac:default_handler:TagsWikiModule components:trac.wiki.web_ui.wikimodule:disabled components:tractags.\*:enabled

}


tractags-env-upgrade(){

    local name=${1:-$SCM_TRAC}
    local env=$SCM_FOLD/tracs/$name

    sudo trac-admin $env upgrade

}





