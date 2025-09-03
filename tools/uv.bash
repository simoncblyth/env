# === func-gen- : tools/uv fgp tools/uv.bash fgn uv fgh tools src base/func.bash
uv-source(){   echo ${BASH_SOURCE} ; }
uv-edir(){ echo $(dirname $(uv-source)) ; }
uv-ecd(){  cd $(uv-edir); }
uv-dir(){  echo $LOCAL_BASE/env/tools/uv ; }
uv-cd(){   cd $(uv-dir); }
uv-vi(){   vi $(uv-source) ; }
uv-env(){  elocal- ; }
uv-usage(){ cat << EOU


uv : faster pip/virtualenv built in rust
===========================================

https://github.com/astral-sh/uv

https://www.datacamp.com/tutorial/python-uv

https://medium.com/@datagumshoe/using-uv-and-conda-together-effectively-a-fast-flexible-workflow-d046aff622f0


zeta/lch installed UV into conda home env
-------------------------------------------



EOU
}
uv-get(){
   local dir=$(dirname $(uv-dir)) &&  mkdir -p $dir && cd $dir

}
