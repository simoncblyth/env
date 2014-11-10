# === func-gen- : base/compression/blosc/cblosc fgp base/compression/blosc/cblosc.bash fgn cblosc fgh base/compression/blosc
cblosc-src(){      echo base/compression/blosc/cblosc.bash ; }
cblosc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cblosc-src)} ; }
cblosc-vi(){       vi $(cblosc-source) ; }
cblosc-env(){      elocal- ; }
cblosc-usage(){ cat << EOU

C-BLOSC : C API to Blosc compression
======================================


blz : extensable ndarray 
---------------------------------

* https://github.com/ContinuumIO/blz


EOU
}
cblosc-dir(){ echo $(local-base)/env/base/compression/c-blosc ; }
cblosc-cd(){  cd $(cblosc-dir); }
cblosc-mate(){ mate $(cblosc-dir) ; }
cblosc-get(){
   local dir=$(dirname $(cblosc-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/Blosc/c-blosc

}
