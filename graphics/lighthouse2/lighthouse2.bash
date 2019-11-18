lighthouse2-source(){   echo ${BASH_SOURCE} ; }
lighthouse2-edir(){ echo $(dirname $(lighthouse2-source)) ; }
lighthouse2-ecd(){  cd $(lighthouse2-edir); }
lighthouse2-dir(){  echo $LOCAL_BASE/env/graphics/lighthouse2/lighthouse2 ; }
lighthouse2-cd(){   cd $(lighthouse2-dir); }
lighthouse2-vi(){   vi $(lighthouse2-source) ; }
lighthouse2-env(){  elocal- ; }
lighthouse2-usage(){ cat << EOU


LightHouse 2
===============

OptiX based renderer from author of Brigade renderer Jacco Bikker.
Brigade was subsequently bought by OTOY.

* https://jacco.ompf2.com/


EOU
}
lighthouse2-get(){
   local dir=$(dirname $(lighthouse2-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d lighthouse2 ] && git clone git@github.com:simoncblyth/lighthouse2.git


}
