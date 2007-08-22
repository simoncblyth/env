traclxml-place-plugins(){

   iwd=$(pwd)
     
   local name=${1:-$SCM_TRAC}
   local fold=$SCM_FOLD/tracs/$name
   [ -d "$fold" ] || ( echo  error no folder $fold && exit 1 )

   cd $LOCAL_BASE/trac/plugins || ( echo error no plugins folder && return 1 ) 
   cd traclxml
  
   echo === name $name fold $fold === 
   echo === copying macros into plugins folder  not wiki-macros as you might expect ===
   sudo -u $APACHE2_USER cp -f traclxml.py $fold/plugins/
    
   cd $iwd  

}


traclxml-config(){

   iwd=$(pwd)
     
   local name=${1:-$SCM_TRAC}
   local fold=$SCM_FOLD/tracs/$name
   [ -d "$fold" ] || ( echo  error no folder $fold && return  1 )

   local path=/Users/blyth/hh/access/rest/identity.xsl

   ini-edit $fold/conf/trac.ini traclxml:stylesheet:$path


}