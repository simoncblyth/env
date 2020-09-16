# === func-gen- : batch/slurm fgp batch/slurm.bash fgn slurm fgh batch src base/func.bash
slurm-source(){   echo ${BASH_SOURCE} ; }
slurm-edir(){ echo $(dirname $(slurm-source)) ; }
slurm-ecd(){  cd $(slurm-edir); }
slurm-dir(){  echo $LOCAL_BASE/env/batch/slurm ; }
slurm-cd(){   cd $(slurm-dir); }
slurm-vi(){   vi $(slurm-source) ; }
slurm-env(){  elocal- ; }
slurm-usage(){ cat << EOU

* https://docs-dev.nersc.gov/cgpu/
* https://docs-dev.nersc.gov/cgpu/usage/
* https://www.nersc.gov/systems/perlmutter/


EOU
}
slurm-get(){
   local dir=$(dirname $(slurm-dir)) &&  mkdir -p $dir && cd $dir

}
