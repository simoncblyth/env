
nuwa-env(){ 
   elocal- 
   local msg="=== $FUNCNAME :"
   local v=$(nuwa-version $*)
   if [ "$(nuwa-isinstalled $*)" == "NO" ]; then
      [ "$v" == "trunk" ] && echo $msg ABORT trunk is not installed && return 1
      echo $msg WARNING nuwa IS NOT installed based on "$*" OR NUWA_HOME : $NUWA_HOME , attempt fallback to trunk 
      nuwa- trunk
   else
      nuwa-functions $*
      nuwa-exports $*   
   fi

   
   
  
}


nuwa-usage(){
  cat << EOU

   Versioning and placement control based on a single variable : NUWA_HOME

           NUWA_HOME              :  $NUWA_HOME
           nuwa-home-default      :  $(nuwa-home-default)

   
   If defined the NUWA_HOME must follow a standard layout for its last 2 levels, eg 
        /whatever/prefix/is/desired/1.0.0-rc02/NuWa-1.0.0-rc02
   if not defined a default is used


   Usage :
            nuwa-functions <v>
                source the dyb__.sh and slave.bash from for version v, defaulting to trunk 
                (equivalent to the former dyb-hookup )

            nuwa-info <v>
                dump the distro parameters for version v, defaulting to trunk 

            nuwa-exports <v>
                export convenience variables
                NB NUWA_HOME is never exported, that is regarded as an input only 

       
   Example of usage in .bash_profile :
   
        ## defines the nuwa- precursor , so must come before that 
        env-     
        
        ## comment the below two lines to use the default of "trunk"
        export NUWA_VERSION=1.0.0-rc02 ... used by dyb__version function for dybinst option setting 
        export NUWA_HOME=whatever/$NUWA_VERSION/NuWa-$NUWA_VERSION   ## NB prior to invoking "nuwa-" precursor 
        
        ## defines the functions and exports  
        nuwa-       
             
        ## setup a default location for operations : building/testing      
        export BUILD_PATH=dybgaudi/${nuwa_version:-trunk}/RootIO/RootIOTest
             
             
             
    To temporarily jump into another release :
        
        nuwa- 1.0.0-rc01          ( OR  trunk ) 
        nuwa-info    
        
     ## NB the dynamics will still be based using the version from NUWA_HOME, but the 
        exports and function paths should now show they hail from the chosen release
        

EOU

   nuwa-info


}

nuwa-dump(){

   nuwa-info
   nuwa-info trunk
   nuwa-info 1.0.0-rc01
   nuwa-info 1.0.0-rc02

}



nuwa-info(){
  local v=$1
  cat << EOI
  
   Dynamically derived quantities for version provided $v 
   If no version argument is given determine the quantities based
   in the value of  NUWA_HOME : $NUWA_HOME 
   
   NB unfortunately dyb__* functions have some dependency on NUWA_VERSION too
       NUWA_VERSION : $NUWA_VERSION  
   
   
   OR a default if that is not defined 
  
          nuwa-home $v    :  $(nuwa-home $v)
          nuwa-release $v :  $(nuwa-release $v)
          nuwa-version $v :  $(nuwa-version $v)
          nuwa-base $v    :  $(nuwa-base $v) 
          nuwa-scripts $v :  $(nuwa-scripts $v)  
          nuwa-dyb__ $v   :  $(nuwa-dyb__ $v)   
          nuwa-slave $v   :  $(nuwa-slave $v)  
  
          nuwa-ddr $v     :  $(nuwa-ddr $v)       
          nuwa-ddi $v     :  $(nuwa-ddi $v)        
          nuwa-ddt $v     :  $(nuwa-ddt $v)       
  
    Exported into environment :
        
            DDR : $DDR
            DDI : $DDI
            DDT : $DDT
            
            
    Source paths reported by the functions hooked up into the environment :
    
         dyb__source : $(dyb__source)
         slave-path  : $(slave-path) 
            
    Checking to see if nuwa for version \"$v\" is installed already based on existance of the dyb__.sh      
  
         nuwa-isinstalled $v : $(nuwa-isinstalled $v)
  
EOI
}




nuwa-home-default(){  echo $LOCAL_BASE/dyb/trunk_dbg/NuWa-trunk ; }

nuwa-home-construct(){
   local v=$1
   case $v in
      trunk) nuwa-home-default ;;
          *) echo $LOCAL_BASE/dyb/releases/$1/NuWa-$1 ;;
   esac       
}

nuwa-home(){          
  if [ "$1" == "" ]; then
     echo ${NUWA_HOME:-$(nuwa-home-default)}
  else 
     nuwa-home-construct $1 
  fi
}

nuwa-release(){       echo $(basename $(nuwa-home $*)); }
nuwa-version(){       local rel=$(nuwa-release $*) ; echo ${rel/NuWa-/} ; }
nuwa-scripts(){       echo installation/$(nuwa-version $*)/dybtest/scripts ; }
nuwa-base(){          echo $(dirname $(nuwa-home $*)) ; } 
nuwa-dyb__(){         echo $(nuwa-base $*)/$(nuwa-scripts $*)/dyb__.sh ; }
nuwa-slave(){         echo $(nuwa-base $*)/$(nuwa-scripts $*)/slave.bash ; }

nuwa-ddr(){           echo $(nuwa-home $*) ; } 
nuwa-ddi(){           echo $(nuwa-base $*)/installation/$(nuwa-version $*)/dybinst/scripts ; }
nuwa-ddt(){           echo $(nuwa-base $*)/installation/$(nuwa-version $*)/dybtest ; }


nuwa-exports(){
   export DDT=$(nuwa-ddt $*)
   export DDI=$(nuwa-ddi $*)
   export DDR=$(nuwa-ddr $*)
}

nuwa-functions(){     
   
    local msg="=== $FUNCNAME : "
    local dyb__=$(nuwa-dyb__ $*)
    local slave=$(nuwa-slave $*)
    
    if [ -f $dyb__ ]; then
        set --
        . $dyb__
        [ -z $BASH_SOURCE ] && eval "function dyb__source(){  echo $dyb__ ; }"      ## workaround for older bash  
        dyb__default(){ echo dybgaudi/Simulation/GenTools ; } 
    else
        echo $msg no dyb__ $dyb__
    fi 
     
    if [ -f $slave ]; then
       . $slave
    else
       echo $msg no slave $slave 
    fi


}

nuwa-isinstalled(){
   nuwa-isinstalled- $* && echo YES || echo NO
}

nuwa-isinstalled-(){
    local dyb__=$(nuwa-dyb__ $*)
    [ -f "$dyb__" ] && return 0 || return 1
}
