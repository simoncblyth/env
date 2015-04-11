# === func-gen- : graphics/ppmfast/ppmfast fgp graphics/ppmfast/ppmfast.bash fgn ppmfast fgh graphics/ppmfast
ppmfast-src(){      echo graphics/ppmfast/ppmfast.bash ; }
ppmfast-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ppmfast-src)} ; }
ppmfast-vi(){       vi $(ppmfast-source) ; }
ppmfast-env(){      elocal- ; }
ppmfast-usage(){ cat << EOU

Fast? Parsing of PPM files
=============================

* http://josiahmanson.com/prose/optimize_ppm/

See also ppm-




EOU
}
ppmfast-dir(){ echo $(env-home)/graphics/ppmfast ; }
ppmfast-cd(){  cd $(ppmfast-dir); }
