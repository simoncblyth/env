# === func-gen- : network/openvpn fgp network/openvpn.bash fgn openvpn fgh network
openvpn-src(){      echo network/openvpn.bash ; }
openvpn-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openvpn-src)} ; }
openvpn-vi(){       vi $(openvpn-source) ; }
openvpn-env(){      elocal- ; }
openvpn-usage(){ cat << EOU

OPENVPN
=========



EOU
}
openvpn-dir(){ echo $(local-base)/env/network/network-openvpn ; }
openvpn-cd(){  cd $(openvpn-dir); }
openvpn-mate(){ mate $(openvpn-dir) ; }
openvpn-get(){
   local dir=$(dirname $(openvpn-dir)) &&  mkdir -p $dir && cd $dir

}
