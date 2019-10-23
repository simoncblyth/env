nest-source(){   echo ${BASH_SOURCE} ; }
nest-edir(){ echo $(dirname $(nest-source)) ; }
nest-ecd(){  cd $(nest-edir); }
nest-dir(){  echo $LOCAL_BASE/env/scintillation/nest/nest ; }
nest-cd(){   cd $(nest-dir); }
nest-vi(){   vi $(nest-source) ; }
nest-env(){  elocal- ; }
nest-usage(){ cat << EOU

Noble Element Simulation Technique
====================================

* http://nest.physics.ucdavis.edu/
* https://solid.physics.ucdavis.edu/pipermail/nest/

* http://inspirehep.net/record/913031/citations?ln=en
* ~/opticks_refs/nest_1106.1613.pdf



EOU
}
nest-get(){
   local dir=$(dirname $(nest-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d nest ] && git clone https://github.com/NESTCollaboration/nest   

   cd nest 
   #git checkout tags/v2.0.0 -b master

}
