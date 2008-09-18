
tracinter-usage(){
   cat << EOU
   
      tracinter-triplets-  : emit abbrev / tag / urls 
         
$(tracinter-triplets-) 
   
   
      tracinter-triplets :  emits the intertrac configuration triplets 
             
$(tracinter-triplets) 
   
   
      tracinter-conf     : 
          
         apply config triplets to TRAC_INSTANCE : $TRAC_INSTANCE
      
        
     Usage may require SUDO / TRAC_INSTANCE overrides :
     
     Old way with override issues, to to resetting in elocal-     
          TRAC_INSTANCE=env SUDO=sudo tracinter-conf
     
     
          trac-
          TRAC_INSTANCE=env SUDO=sudo trac-intertrac-conf
   
   
     Check the config by visiting wiki:InterTrac on the target instance
   
   
   
EOU

}

tracinter-env(){
  echo -n
}




tracinter-triplet(){

   local abbrev=$1
   local tag=$2
   local url=$3
   local name=$(tracinter-url2name $url)
   local title
   [ "$name" != "$tag" ] && title=${tag}_${name}_instance || title=${tag}_instance 

cat << EOI
     intertrac:$abbrev:$tag 
     intertrac:$tag.compat:false  
     intertrac:$tag.title:$title 
     intertrac:$tag.url:$url
EOI

}


tracinter-url2name(){
  case $1 in 
     http://bitten.edgewall.org) echo bitten ;;
       http://trac.edgewall.org) echo trac ;;
          http://trac-hacks.org) echo trachacks ;;
                              *) echo $(basename $1) ;;
  esac
}

tracinter-triplets-(){
   cat << EOT
   i  ihep      http://dayabay.ihep.ac.cn/tracs/dybsvn
   e  env       http://dayabay.phys.ntu.edu.tw/tracs/env
   m  mirror    http://dayabay.phys.ntu.edu.tw/tracs/dybsvn
   b  bitten    http://bitten.edgewall.org
   tr trac      http://trac.edgewall.org
   th trachacks http://trac-hacks.org
   t  tracdev   http://dayabay.phys.ntu.edu.tw/tracs/tracdev
EOT

if [ "$NODE_TAG" != "XX" ]; then
   cat << EOT 
   w  workflow     http://localhost/tracs/workflow
   bn bittennotify http://trac.3dbits.de/bittennotify
   gs genshi       http://genshi.edgewall.org
EOT
fi

}


tracinter-triplets(){
   local triplet
   tracinter-triplets- | while read triplet ; do
      tracinter-triplet $triplet
   done
}



tracinter-conf(){
   echo $msg DEPRECATED USE trac-intertrac-conf due to SUDO complications 
   local name=${1:-$TRAC_INSTANCE}
   trac-
   SUDO=$SUDO TRAC_INSTANCE=$name trac-configure $(tracinter-triplets)
}



