
trac2latex-get(){

   #  documented at 
   # http://trac-hacks.org/wiki/Trac2LatexPlugin
   #

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins
    
   svn co http://trac2latex.googlecode.com/svn/trunk/ trac2latex
   cd trac2latex

}

trac2latex-install(){

    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd trac2latex/0.10/plugins
    python setup.py install 
      
    #  Installed /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/TracTrac2Latex-0.0.1-py2.5.egg
    #  
    #python setup.py bdist_egg
}

trac2latex-place-macros(){
 
     local name=${1:-dummy}
     local fold=$SCM_FOLD/tracs/$name
     [ -d "$fold" ] || ( echo  error no folder $fold && exit 1 )

     cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
     cd trac2latex/0.10
  
     sudo -u $APACHE2_USER cp -f wiki-macros/* $fold/wiki-macros/ 

[g4pb:/var/scm/tracs/workflow] blyth$ sudo -u www mv wiki-macros/*.py plugins/
Password:
[g4pb:/var/scm/tracs/workflow] blyth$ sudo -u www mv plugins/formula.py wiki-macros/


}


trac2latex-enable(){

   name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:trac2latex.\*:enabled

}