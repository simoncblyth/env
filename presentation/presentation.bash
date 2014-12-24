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


presentation-name(){ echo gpu_accelerated_geant4_simulation ; }
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
 


