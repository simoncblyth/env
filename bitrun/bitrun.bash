

bitrun-rc(){ echo $HOME/.bitrunrc ; }

bitrun-usage(){

   . $(bitrun-rc)
      
cat << EOU

   For example hookup to your environment and invoke with
       . path/to/bitrun.bash && bitrun-start

   These bitrun-* functions provide the standard ways of invoking 
   the bitten-slave, to preform automated tests/builds.

    bitrun-rc       : $(bitrun-rc)
              absolute path to basic configuration file, which should be read protected 
    
    == basic config parameters ==
    
    bitrun-name     : $(bitrun-name)
    bitrun-url      : $(bitrun-url)
              url of the controlling "master" trac instance with which to communicate          
     
    bitrun-cfg      : $(bitrun-cfg)
              absolute path to the configuration file of the local slave node, exactly what should
              appear in the depends on the recipes that this slave is going to cook ... could 
              contain things such as the absolute path to the repository 
    
    == checks ==  
       
    which bitten-slave : $(which bitten-slave)
              if the above is blank then you need to install nosebit and get python
              into you path     
              
    == commands ==

    bitrun-start :
              start the slave , it will contact the master to find if there are any pending
              tasks that this slave is able to perform
      
    == notes ==


     Bitten slave option notes :    
          
              the build-dir is by default created within the work-dir with a name like 
              build_\${config}_\${build}   setting it to "" as used here  is a convenience for testing
              which MUST go together with "--keep-files" to avoid potentially deleting bits 
              of working copy      
    
    Recipe tips :
    
       -  shield the slave from non-zero return codes with "echo \$?" for example 
       -  escaping backslashes in xml is problematic, why ?
              when doing \${p/trunk\//} it somehow becomes  \${p/trunk\\//} which doesnt work
              avoid the issue by using   .\${p/trunk/} 
       - be aware of the different directories in use
             - the invokation context in /tmp/env/bitrun-start/etc..
             - it seems that sh:exec gets along fine with that ... or pehaps its just capturing stdout and ignoring 
               the files it is creating ???
               ... but the explicit report needs a path 
               
                                                                                                                                                                                                                                                                                           
EOU
}

  

  
bitrun-check(){
   . $(bitrun-rc)
   local msg="=== $FUNCNAME :"
   [ -z $name ] && echo $msg no name && return 1
   [ -z $url  ] && echo $msg no url && return 1
   [ -z $user ] && echo $msg no user  && return 1
   [ -z $pass ] && echo $msg no pass && return 1
   [ -z $cfg  ] && echo $msg no cfg   && return 1
   return 0
}
 
bitrun-name(){ . $(bitrun-rc) ; echo $name ; } 
bitrun-url(){  . $(bitrun-rc) ; echo $url ; }
bitrun-cfg(){  . $(bitrun-rc) ; echo $cfg ; } 
 
bitrun-start(){

    . $(bitrun-rc)
    local msg="=== $FUNCNAME :"
    [ "$(which bitten-slave)" == "" ] && echo $msg no bitten-slave in your path  && return 1
    ! bitrun-check      && echo $msg ABORT create a $HOME/.bitrunrc with the needed config && return 1
    [ ! -f $cfg ]       && echo $msg ERROR cfg file $cfg does not exist && return 1 

    local log=${FUNCNAME/-*}.log

    local iwd=$PWD
    local tmp=$(bitrun-dir) && mkdir -p $tmp
    cd $tmp
    local cmd="bitten-slave --verbose --config=$cfg --dump-reports --work-dir=. --build-dir=  --keep-files $* --user=$user --password=$pass --log=$log  $url"
    #echo $cmd
    eval "$cmd"  
    cd $iwd
}


bitrun-dir(){
   . $(bitrun-rc)
   local tmp=/tmp/$name/${FUNCNAME/-*/} 
   echo $tmp
}

bitrun-cd(){
  cd $(bitrun-dir)
}






