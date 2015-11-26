# === func-gen- : physics/refractiveindex/refractiveindex fgp physics/refractiveindex/refractiveindex.bash fgn refractiveindex fgh physics/refractiveindex
refractiveindex-src(){      echo physics/refractiveindex/refractiveindex.bash ; }
refractiveindex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(refractiveindex-src)} ; }
refractiveindex-vi(){       vi $(refractiveindex-source) ; }
refractiveindex-env(){      elocal- ; }
refractiveindex-usage(){ cat << EOU




EOU
}
refractiveindex-dir(){ echo $LOCAL_BASE/env/physics/refractiveindex ; }
refractiveindex-edir(){ echo $(env-home)/physics/refractiveindex ; }
refractiveindex-ecd(){  cd $(refractiveindex-edir); }
refractiveindex-cd(){   cd $(refractiveindex-dir); }


