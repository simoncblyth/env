

trac2mediawiki-get(){

   #  documented at 
   #    http://localhost/tracs/workflow/wiki/Trac2MediaWikiDevelopment
   #

   cd $LOCAL_BASE/trac
   [ -d "plugins" ] || mkdir -p plugins
   cd plugins
    
   svn co http://dayabay.phys.ntu.edu.tw/repos/trac2mediawiki/trunk/ trac2mediawiki
   cd trac2mediawiki

}



trac2mediawiki-install(){

    iwd=$(pwd)
     
    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd trac2mediawiki/0.10/plugins
    python setup.py install 
    sudo apachectl restart
        
    cd $iwd  
}

trac2mediawiki-place-macros(){
 
     local name=${1:-dummy}
     local fold=$SCM_FOLD/tracs/$name
     [ -d "$fold" ] || ( echo  error no folder $fold && exit 1 )

     cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
     cd trac2mediawiki/0.10
  
     sudo -u $APACHE2_USER cp -f wiki-macros/* $fold/plugins/
   # sudo -u $APACHE2_USER cp -f wiki-macros/* $fold/wiki-macros/ 

# [g4pb:/var/scm/tracs/workflow] blyth$ sudo -u www mv wiki-macros/*.py plugins/
# Password:
# [g4pb:/var/scm/tracs/workflow] blyth$ sudo -u www mv plugins/formula.py wiki-macros/

}


trac2mediawiki-enable(){

   name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:trac2mediawiki.\*:enabled

}