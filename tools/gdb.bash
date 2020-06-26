# === func-gen- : tools/gdb fgp tools/gdb.bash fgn gdb fgh tools src base/func.bash
gdb-source(){   echo ${BASH_SOURCE} ; }
gdb-edir(){ echo $(dirname $(gdb-source)) ; }
gdb-ecd(){  cd $(gdb-edir); }
gdb-dir(){  echo $LOCAL_BASE/env/tools/gdb ; }
gdb-cd(){   cd $(gdb-dir); }
gdb-vi(){   vi $(gdb-source) ; }
gdb-env(){  elocal- ; }
gdb-usage(){ cat << EOU

GDB
====

Refs
-----

* https://sourceware.org/gdb/onlinedocs/gdb/Set-Breaks.html



Avoid the annoying quit confirmation
--------------------------------------

* https://stackoverflow.com/questions/4355978/get-rid-of-quit-anyway-prompt-using-gdb-just-kill-the-process-and-quit

~/.gdbinit::

   define hook-quit
      set confirm off
   end


Avoid confirming pending breakpoints
---------------------------------------

::

    set breakpoint pending on



Scripted setting of multiple breakpoints
------------------------------------------

::

     44 tds(){
     ..
     59    local script=$JUNOTOP/offline/Examples/Tutorial/share/tut_detsim.py
     60    local args="--opticks --no-guide_tube gun"
     61    #local args="--opticks gun"  
     62 
     63    if [ -z "$BP" ]; then
     64       H="" 
     65       B=""         
     66       T="-ex r"
     67    else
     68       H="-ex \"set breakpoint pending on\" "
     69       B="" 
     70       for bp in $BP ; do
     71          B="$B -ex \"break $bp\" "
     72       done
     73       T="-ex \"info break\" -ex r"
     74    fi 
     75    
     76    local runline="gdb $H $B $T --args python $script $args $* "
     77    echo $runline
     78    eval $runline
     79 }  
     80 
     81 tds0(){ BP=DetSim0Svc::createDetectorConstruction tds ; }
     82 tds1(){ BP=R12860PMTManager::R12860PMTManager     tds ; }
     83 tds2(){ BP="R12860PMTManager::R12860PMTManager DetSim0Svc::createDetectorConstruction"   tds ; }
     84 




Breakpoints
--------------

* plant a std::raise(SIGINT) at strategic point where symbols will be available, 
  and switch that on with an option, eg --cg4sigint 

* TAB completion of symbolic bp requires all namespaces 

::

    (gdb) b 'HepRandom::put'
    Function "HepRandom::put" not defined.
    Make breakpoint pending on future shared library load? (y or [n]) n


    (gdb) b "CLHEP::HepRandom::put( <TAB>    DOES NOT WORK WITH DOUBLE QUOTES        

    (gdb) b 'CLHEP::HepRandom::put( <TAB>    THIS WORKS : BUT FINDS WRONG SYMBOL

    (gdb) b 'CLHEP::HepRandom::put(std::ostream&) const' 



    (gdb) b /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc:58
    Breakpoint 1 at 0x7fffe7b4d46e: file /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/externals/clhep/src/RandomEngine.cc, line 58.





EOU
}
gdb-get(){
   local dir=$(dirname $(gdb-dir)) &&  mkdir -p $dir && cd $dir

}
