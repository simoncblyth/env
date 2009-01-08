
cernlib-source(){ echo $BASH_SOURCE ; }
cernlib-usage(){  
  cat << EOU
  
      http://cernlib.web.cern.ch/cernlib/
      http://cernlib.web.cern.ch/cernlib/install/start_cern
      http://cernlib.web.cern.ch/cernlib/download/2006_source/tar/
      
      issues :
          gfortran cf g77
          
      see #i151    
          
      
      
EOU
}


cernlib-env(){
  echo -n
}

