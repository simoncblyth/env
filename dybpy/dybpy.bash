

dybpy-env(){
  elocal-
  dyb-
}

dybpy-usage(){

cat  << EOU

  which python : $(which python)

  dybpy-setup :   
      invoke setup in develop mode 
      providing a beachhead into NuWa python sys.path via an egglink 
      ... only needs to be done once
     
  dybpy-rsync :
      for under the svn radar transfers to other nodes
      
  dybpy-envtest :
      syspath dumping 
      
      
  dybpy-cmd "<python commands>"  options... :
       run python -c commands from a temporary directory, avoiding accidental module pickups
  
  dybpy-icmd "<python commands>" options... :
       create a temporary file from the python commands given and invoke with 
       ipython, dropping into interactive mode with objects you created in the commands
       ready to play with 
       
       The temporary file is created by converting ";" to "\n" 
       for example :
             'from dybpy import * ;gel = GenEventLook() ;gel.run() ;'
    
        NB there are no spaces after the ";" to prevent bad indentation 
   
        Noting slow startup speed for ipython ... circa 10 seconds
        when using "--quick" option skipping reading config it takes 2 seconds  
   
        Maybe due to heavy sys.path ?
   
   
   dybpy-gel <i>
        an example of a dybpy-*cmd usage , goes from scratch to ipython commandline
        with your objects ready to play with
   
   


EOU

}

dybpy-rsync(){  env-rsync ${FUNCNAME/-*} ${1:-P} ;  }

dybpy-setup(){
  local iwd=$PWD 
  cd $ENV_HOME/dybpy
  python setup.py develop
  cd $iwd
}



dybpy-envtest(){
   local msg="=== $FUNCNAME : "
   local paths="dybgaudi/DybRelease dybgaudi/Simulation/GenTools "
   for p in $paths
   do
       echo $msg $p
       dyb__ $p  > /dev/null
       python -c "import dybpy as dp ; dp.syspath()  "
   done
}


dybpy-cmd(){
    local cmd=$1
    shift
    local iwd=$PWD
    local tmp=/tmp/env/${FUNCNAME/-*} && mkdir -p $tmp
    dyb__
    cd $tmp        
    python $* -c $cmd       
}

dybpy-icmd(){
    local msg="=== $FUNCNAME :"
    local cmd=$1
    shift
    
    local iwd=$PWD
    local tmp=/tmp/env/${FUNCNAME/-*} && mkdir -p $tmp
    local nam=tmp.py
    
    echo $msg setting up cmt controlled environment 
    time dyb__
    cd $tmp
    
    ## keep the decks clean to prevent globbing ... but didnt 
    [ -f $nam ] && rm -f $nam        
               
    ## quoting needed to prevent globbing                                            
    echo "$cmd" | tr ";" "\n" > $nam
    echo $msg created temporary $nam
    cat $nam
    
    echo $msg entering ipython with options [$*]
    ipython $* tmp.py


    cd $iwd
}



dybpy-gel(){
   local i=${1:-i}
   local cmd="dybpy-${i}cmd 'from dybpy import * ;gel = GenEventLook() ;gel.run() ;esv=gel.esv ;app=gel.app ;'"
   ## anti glob quotes
   echo "$cmd"
   eval "$cmd"
}




