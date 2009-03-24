tracpreq-src(){   echo trac/tracpreq.bash ; }
tracpreq-source(){ echo ${BASH_SOURCE:-$(env-home)/$(tracpreq-src)} ; }
tracpreq-vi(){     vi $(tracpreq-source) ; }
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


   env-initlog $FUNCNAME   

   python- 
   pythonbuild-       
   pythonbuild-again   | env-log $FUNCNAME pythonbuild-again
   
   configobj-          
   configobj-get       | env-log $FUNCNAME configobj-get 
   
   
   swig-               
   swigbuild-           
   swigbuild-again     | env-log $FUNCNAME swigbuild-again
   
   apache-
   apache-again        | env-log $FUNCNAME apache-again
   
   svn-
   svnbuild-
   svnbuild-again      | env-log $FUNCNAME svnbuild-again

   sqlite-
   sqlite-again        | env-log $FUNCNAME sqlite-again


}

