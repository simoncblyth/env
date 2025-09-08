# === func-gen- : tools/fastapi fgp tools/fastapi.bash fgn fastapi fgh tools src base/func.bash
fastapi-source(){   echo ${BASH_SOURCE} ; }
fastapi-edir(){ echo $(dirname $(fastapi-source)) ; }
fastapi-ecd(){  cd $(fastapi-edir); }
fastapi-dir(){  echo $LOCAL_BASE/env/tools/fastapi ; }
fastapi-cd(){   cd $(fastapi-dir); }
fastapi-vi(){   vi $(fastapi-source) ; }
fastapi-env(){  elocal- ; }
fastapi-usage(){ cat << EOU


Setup::

    cd /usr/local/env
    mkdir fastapi_check
    cd fastapi_check

    uv venv
    uv pip install "fastapi[standard]"
    uv pip install numpy




EOU
}
fastapi-get(){
   local dir=$(dirname $(fastapi-dir)) &&  mkdir -p $dir && cd $dir

}
