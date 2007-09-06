
tractags-get(){

   # http://www.trac-hacks.org/wiki/TagsPlugin

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins

   local url=http://trac-hacks.org/svn/tagsplugin/tags/0.4.1/
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
  
   local name=${1:-$SCM_TRAC}
   echo === tractags-conf CAUTION changing the trac:default_handler:TagsWikiModule from WikiModule
   
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini trac:default_handler:TagsWikiModule components:trac.wiki.web_ui.wikimodule:disabled components:tractags.\*:enabled

}


tractags-env-upgrade(){

    local name=${1:-$SCM_TRAC}
    local env=$SCM_FOLD/tracs/$name

    sudo trac-admin $env upgrade

}





