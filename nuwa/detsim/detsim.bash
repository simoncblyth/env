# === func-gen- : nuwa/detsim/detsim fgp nuwa/detsim/detsim.bash fgn detsim fgh nuwa/detsim
detsim-src(){      echo nuwa/detsim/detsim.bash ; }
detsim-source(){   echo ${BASH_SOURCE:-$(env-home)/$(detsim-src)} ; }
detsim-vi(){       vi $(detsim-source) ; }
detsim-env(){      elocal- ; dybinst- ; }
detsim-usage(){ cat << EOU





EOU
}
detsim-dir(){ echo $(dybinst-dir)/NuWa-trunk/dybgaudi/Simulation/DetSim ; }
detsim-cd(){  cd $(detsim-dir)/$1; }

detsim-sdir(){ echo $(env-home)/nuwa/detsim ; }
detsim-scd(){ cd $(detsim-sdir)/$1 ; }

detsim-mate(){ mate $(detsim-dir) ; }
detsim-get(){
   local dir=$(dirname $(detsim-dir)) &&  mkdir -p $dir && cd $dir

}




