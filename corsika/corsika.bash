# === func-gen- : corsika/corsika fgp corsika/corsika.bash fgn corsika fgh corsika src base/func.bash
corsika-source(){   echo ${BASH_SOURCE} ; }
corsika-edir(){ echo $(dirname $(corsika-source)) ; }
corsika-ecd(){  cd $(corsika-edir); }
corsika-dir(){  echo $LOCAL_BASE/env/corsika/corsika ; }
corsika-cd(){   cd $(corsika-dir); }
corsika-vi(){   vi $(corsika-source) ; }
corsika-env(){  elocal- ; }
corsika-usage(){ cat << EOU



Towards a Next Generation of CORSIKA: A Framework for the Simulation of
Particle Cascades in Astroparticle Physics

* https://arxiv.org/abs/1808.08226
* https://arxiv.org/pdf/1808.08226.pdf


EOU
}
corsika-get(){
   local dir=$(dirname $(corsika-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://gitlab.ikp.kit.edu/AirShowerPhysics/corsika.git 

}

corsika-f(){  corsika-cd ; find . -type f -exec grep -H ${1:-gdml} \;  ; }
