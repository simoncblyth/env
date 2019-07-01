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
