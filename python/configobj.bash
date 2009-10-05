
configobj-source(){ echo ${BASH_SOURCE:-$(env-home)/python/configobj.bash} ; }
configobj-vi(){     vi $(configobj-source) ; }
configobj-env(){
   elocal-
}
configobj-dir(){ echo $(local-base)/env/configobj ; }
configobj-cd(){  cd $(configobj-dir); }
configobj-mate(){ mate $(configobj-dir) ; }
configobj-usage(){
   cat << EOU
   
     http://www.voidspace.org.uk/python/configobj.html
         
      Enhanced manipulation of ini configuration files
                        
         configobj-get
              easy install into python


 
         configobj-check
               try to import 


     Beware of the sudo python chestnut :
         /data/env/system/python/Python-2.5.1/bin/python: 
            error while loading shared libraries: libpython2.5.so.1.0: cannot open shared object file: No such file or directory


EOU
}

configobj-build(){
  
   local msg="=== $FUNCNAME :"
   configobj-get
   ! configobj-check && echo $msg FAILED sleeping && sleep 1000000000000
}


configobj-install(){
   python-
   local cmd="sudo easy_install configobj "
   echo $cmd 
   eval $cmd
}

configobj-check(){ python -c "import configobj" ; }
configobj-version(){ python -c "import configobj as _ ; print _.__version__" ; }

configobj-url(){ echo http://configobj.googlecode.com/svn/trunk ; }
configobj-get(){
   local dir=$(dirname $(configobj-dir)) && mkdir -p $dir && cd $dir
   svn co $(configobj-url) configobj  
}





