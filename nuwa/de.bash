# === func-gen- : nuwa/de fgp nuwa/de.bash fgn de fgh nuwa
de-src(){      echo nuwa/de.bash ; }
de-source(){   echo ${BASH_SOURCE:-$(env-home)/$(de-src)} ; }
de-vi(){       vi $(de-source) ; }
de-env(){      
   elocal-  
   #fenv  
}
de-usage(){ cat << EOU

de : Detector Element Dumping
================================

Based on 

#. NuWa-trunk/dybgaudi/Detector/XmlDetDesc/python/XmlDetDesc/dump_geo.py
 

EOU
}

de-path(){
   case $1 in 
     all) echo /dd ;;
     oil) echo /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1 ;; 
     first) echo /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column1 ;;
     last) echo /dd/Structure/AD/far-oil4/far-ad4-ring0-column6 ;; 
      111) echo /dd/Structure/AD/db-oil1/db-ad1-ring1-column1 ;; 
   esac
}

de-runenv(){
   csa-
   csa-envcache-source
}

de-dump(){

   de-runenv
   local tag=${1:-oil}
   local path=$(de-path $tag)
   nuwa.py -n1 -m "XmlDetDescChecks.dedump $path"
}
de-main(){
   de-dump $*
}



