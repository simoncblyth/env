# === func-gen- : presentation/presentation fgp presentation/presentation.bash fgn presentation fgh presentation
presentation-src(){      echo presentation/presentation.bash ; }
presentation-source(){   echo ${BASH_SOURCE:-$(env-home)/$(presentation-src)} ; }
presentation-vi(){       vi $(presentation-source) ; }
presentation-env(){      elocal- ; }
presentation-usage(){ cat << EOU





EOU
}
presentation-dir(){ echo $(env-home)/presentation ; }
presentation-cd(){  cd $(presentation-dir); }


#presentation-name(){ echo gpu_accelerated_geant4_simulation ; }
presentation-name(){ echo optical_photon_simulation_with_nvidia_optix ; }

presentation-path(){ echo $(presentation-dir)/$(presentation-name).txt ; }
presentation-export(){
   export PRESENTATION_NAME=$(presentation-name)
}
presentation-edit(){ vi $(presentation-path) ; }
presentation-make(){
   presentation-cd
   presentation-export
   env | grep PRESENTATION
   make $*
}


presentation-remote(){
   echo simoncblyth.bitbucket.org
}

presentation-open(){
   open http://localhost/env/presentation/$(presentation-name).html?page=${1:-0}
} 

presentation-open-remote(){
   open http://$(presentation-remote)/env/presentation/$(presentation-name).html?page=${1:-0}
}

