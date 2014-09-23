# === func-gen- : muon_simulation/presentation/presentation fgp muon_simulation/presentation/presentation.bash fgn presentation fgh muon_simulation/presentation
presentation-src(){      echo muon_simulation/presentation/presentation.bash ; }
presentation-source(){   echo ${BASH_SOURCE:-$(env-home)/$(presentation-src)} ; }
presentation-vi(){       vi $(presentation-source) ; }
presentation-env(){      elocal- ; }
presentation-usage(){ cat << EOU





EOU
}
presentation-dir(){ echo $(env-home)/muon_simulation/presentation ; }
presentation-cd(){  cd $(presentation-dir); }
presentation-mate(){ mate $(presentation-dir) ; }
