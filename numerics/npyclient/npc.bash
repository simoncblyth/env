# === func-gen- : numerics/npyclient/npc fgp numerics/npyclient/npc.bash fgn npc fgh numerics/npyclient
npc-src(){      echo numerics/npyclient/npc.bash ; }
npc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(npc-src)} ; }
npc-vi(){       vi $(npc-source) ; }
npc-usage(){ cat << EOU

NPY Client Testing
=====================




EOU
}
npc-env(){      elocal- ; opticks- ; vs- ;  }
npc-dir(){   echo $(env-home)/numerics/npyclient ; }
npc-bdir(){  echo $(opticks-prefix)/build/numerics/npyclient ; }

npc-cd(){    cd $(npc-dir); }
npc-bcd(){   cd $(npc-bdir); }

npc-name(){ echo NPYClient ; }





