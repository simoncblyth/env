# === func-gen- : proj/key4hep/sio fgp proj/key4hep/sio.bash fgn sio fgh proj/key4hep src base/func.bash
sio-source(){   echo ${BASH_SOURCE} ; }
sio-edir(){ echo $(dirname $(sio-source)) ; }
sio-ecd(){  cd $(sio-edir); }
sio-dir(){  echo $LOCAL_BASE/env/proj/key4hep/SIO ; }
sio-cd(){   cd $(sio-dir); }
sio-vi(){   vi $(sio-source) ; }
sio-env(){  elocal- ; }
sio-usage(){ cat << EOU


https://github.com/iLCSoft/SIO

SIO is a persistency solution for reading and writing binary data in SIO
structures called record and block. SIO has originally been implemented as
persistency layer for LCIO.



EOU
}
sio-get(){
   local dir=$(dirname $(sio-dir)) &&  mkdir -p $dir && cd $dir

    [ ! -d "SIO" ] && git clone https://github.com/iLCSoft/SIO

}
