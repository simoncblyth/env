# === func-gen- : muon_simulation/weighted_optical_photon/opw fgp muon_simulation/weighted_optical_photon/opw.bash fgn opw fgh muon_simulation/weighted_optical_photon
opw-src(){      echo muon_simulation/optical_photon_weighting/opw.bash ; }
opw-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opw-src)} ; }
opw-vi(){       vi $(opw-source) ; }
opw-env(){      elocal- ; }
opw-usage(){ cat << EOU

OPW
====

FUNCTIONS
---------

*opw-get*
    checkout my people copy of Davids OPW

*opw-gen-muons*
    10k muon vectors took ~6 seconds to generate, the sample is reproducible::

        [blyth@belle7 OPW]$ diff  tenthousandmuons tenthousandmuons.1 
        [blyth@belle7 OPW]$ 

EOU
}
opw-dir(){ echo $(local-base)/env/muon_simulation/optical_photon_weighting/OPW ; }
opw-cd(){  cd $(opw-dir); }
opw-mate(){ mate $(opw-dir) ; }
opw-get(){
    local dir=$(dirname $(opw-dir)) &&  mkdir -p $dir && cd $dir
    local nam=$(basename $(opw-dir))

    [ -z "$DYBSVN" ] && echo $msg missing DYBSVN && return 1 
    [ ! -d "$nam" ] && svn co $DYBSVN/people/blyth/$nam
    [   -d "$nam" ] && svn up $nam
}

opw-prep(){
    opw-gen-muons
}

opw-gen-muons-smpl(){ echo $(opw-dir)/tenthousandmuons ; }
opw-gen-muons(){
    type $FUNCNAME
    opw-cd
    local smpl=$(opw-gen-muons-smpl)
    [ -f "$smpl" ] &&  echo $msg sample already created $smpl && return 0
   
    time Muon.exe -n 10000 -s DYB -seed 1377339 -r Yes -v RPC -music_dir $SITEROOT/../external/data/0.0/Muon > $smpl
    wc $smpl
    du -hs $smpl
}


opw-chk(){
    [ "$(which nuwa.py 2> /dev/null)" == "" ] && echo $msg setup nuwa environment first laddie && sleep 1000000000
}

opw-tag(){ echo ${OPW_TAG:-226} ; }
opw-sim(){
   type $FUNCNAME
   opw-cd
   opw-chk

   mkdir -p out log 
   local tag=$(opw-tag)
   time nuwa.py -R 3 -n 1000 -m "$(opw-sim-args)" -o out/$tag.root >& log/$tag.log &
   #time nuwa.py -R 3 -n 1000 -m "$(opw-sim-args)" -o out/$tag.root 
}

opw-sim-args(){ cat << EOA
     fmcpmuon --use-pregenerated-muons --use-basic-physics 
        --wsLimit=1 
        --wsWeight=1 
        --adVolumes=['oil','lso','gds'] 
        --adLimits=[1,3000,1000] 
        --adWeights=[1,100,100]

EOA
}


opw-ana(){
   opw-chk
   opw-cd
   local tag=$(opw-tag)
   #nuwa.py -n -1 -m "opa" out/$tag.root >& log/${tag}opa.log
   nuwa.py -n -1 -m "opa" out/$tag.root 
}

opw-plt(){
   opw-chk
   opw-cd
   

}
