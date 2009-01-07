

tracpreq-usage(){

  cat << EOU
  
      tracpreq-again 
            build again the prerequisites for a trac server ...
            CAUTION existing installations ARE WIPED 
            
            ensure any build and install directory customisations 
            are safety tucked away in patches before doing this   
  
EOU

}

tracpreq-env(){
   echo -n
}

tracpreq-again(){

   python-
   pythonbuild-
   pythonbuild-again
   
   configobj-
   configobj-get
   
   
   swig-
   swigbuild-
   swigbuild-again
   
   apache-
   apache-again
   
   svn-
   svnbuild-
   svnbuild-again

   sqlite-
   sqlite-again 


}

