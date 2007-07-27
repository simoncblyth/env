trac-plugin-reposearch-get(){

   #  documented at 
   #  http://www.trac-hacks.org/wiki/RepoSearchPlugin
   #

   local rame=reposearchplugin
   local name=tracreposearch

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins
    
   svn co http://trac-hacks.org/svn/$rame $name
   cd $name

}

trac-plugin-reposearch-install(){

    local name=tracreposearch
    
    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd $name/0.10
    python setup.py install 

#Installing update-index script to /usr/local/python/Python-2.5.1/bin
#Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/tracreposearch-0.2-py2.5.egg
   
}


trac-plugin-reposearch-enable(){

   local name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:tracreposearch.\*:enabled

}

trac-plugin-reposearch-permission(){

   local name=${1:-$SCM_TRAC}
   trac-conf-perm $name add authenticated REPO_SEARCH
   trac-conf-perm $name list 
   
}