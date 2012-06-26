tracincludemacro-usage(){ cat << EOU

   #  documented at 
   #  http://trac-hacks.org/wiki/IncludeMacro

EOU
}

tracincludemacro-get(){


   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins
    
   svn co http://trac-hacks.org/svn/includemacro
   cd includemacro/0.10
   

}

tracincludemacro-install(){

    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd includemacro/0.10
    python setup.py install 
 
  # Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracIncludeMacro-1.0-py2.5.egg
}


tracincludemacro-enable(){

   name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:includemacro.\*:enabled

}
