# === func-gen- : proj/key4hep/podio fgp proj/key4hep/podio.bash fgn podio fgh proj/key4hep src base/func.bash
podio-source(){   echo ${BASH_SOURCE} ; }
podio-edir(){ echo $(dirname $(podio-source)) ; }
podio-ecd(){  cd $(podio-edir); }
podio-dir(){  echo $LOCAL_BASE/env/proj/key4hep/podio ; }
podio-cd(){   cd $(podio-dir); }
podio-vi(){   vi $(podio-source) ; }
podio-env(){  elocal- ; }
podio-usage(){ cat << EOU


https://github.com/AIDASoft


PODIO: recent developments in the Plain Old Data EDM toolkit

    Frank Gaede. DESY
    Benedikt Hegner, CERN
    Graeme A. Stewart, CERN
    2020

https://inspirehep.net/literature/1832163

https://github.com/iLCSoft/SIO


EOU
}
podio-get(){
   local dir=$(dirname $(podio-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d podio ] && git clone https://github.com/AIDASoft/podio

}
