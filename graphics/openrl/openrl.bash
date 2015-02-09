# === func-gen- : graphics/openrl/openrl fgp graphics/openrl/openrl.bash fgn openrl fgh graphics/openrl
openrl-src(){      echo graphics/openrl/openrl.bash ; }
openrl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openrl-src)} ; }
openrl-vi(){       vi $(openrl-source) ; }
openrl-env(){      elocal- ; }
openrl-usage(){ cat << EOU

OpenRL
=======

* http://en.m.wikipedia.org/wiki/OpenRL

PowerVR OpenRL is a flexible low level interactive ray tracing API, available
for download as an SDK for accelerating ray tracing in both graphics and
non-graphics (e.g., physics) applications. OpenRL was developed by the Caustic
Professional division of Imagination Technologies. A free perpetual license of
OpenRL is available for integration, with either commercial or non-commercial
applications.


* from the makers of iPhone/iPad GPUs 

* http://www.anandtech.com/show/7870/imagination-announces-powervr-wizard-gpu-family-rogue-learns-ray-tracing

* http://www.imgtec.com/powervr/raytracing.asp

* https://code.google.com/p/heatray/



EOU
}
openrl-dir(){ echo $(local-base)/env/graphics/openrl/graphics/openrl-openrl ; }
openrl-cd(){  cd $(openrl-dir); }
openrl-mate(){ mate $(openrl-dir) ; }
openrl-get(){
   local dir=$(dirname $(openrl-dir)) &&  mkdir -p $dir && cd $dir

}
