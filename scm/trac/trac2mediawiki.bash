

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


trac2mediawiki-update(){
   
   iwd=$(pwd)

   cd $LOCAL_BASE/trac/plugins/trac2mediawiki
   pwd
   svn info
   svn up 
 
   cd $iwd 
}


trac2mediawiki-install(){

    iwd=$(pwd)
    
    cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
    cd trac2mediawiki/0.10/plugins
    
    echo === installing into site packages of $(which python) ===
    python setup.py install 
    
    echo === restarting apache using $(which apachectl) ===
    sudo apachectl restart
        
    cd $iwd  
}

trac2mediawiki-place-macros(){
 
     iwd=$(pwd)
     
     local name=${1:-$SCM_TRAC}
     local fold=$SCM_FOLD/tracs/$name
     [ -d "$fold" ] || ( echo  error no folder $fold && exit 1 )

     cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
     cd trac2mediawiki/0.10
  
     echo === copying macros into plugins folder  not wiki-macros as you might expect ===
     sudo -u $APACHE2_USER cp -f wiki-macros/* $fold/plugins/
    
     cd $iwd  
}


trac2mediawiki-enable(){

   name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:trac2mediawiki.\*:enabled

}

trac2mediawiki-configure(){

   name=${1:-$SCM_TRAC}
   group=${1:-$SCM_GROUP}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini  trac2mediawiki:scmgroup:$group

}

