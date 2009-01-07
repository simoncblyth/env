
configobj-source(){ echo ${BASH_SOURCE:-$(env-home)/python/configobj.bash} ; }
configobj-env(){
   elocal-
}

configobj-usage(){
   cat << EOU
   
     http://www.voidspace.org.uk/python/configobj.html
         
      Enhanced manipulation of ini configuration files
                        
         configobj-get
              easy install into python
 
         configobj-check
               try to import 
EOU
}

configobj-get(){
   python-
   easy_install configobj 
}

configobj-check(){
   python -c "import configobj"
}