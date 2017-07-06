# === func-gen- : graphics/povray/povray fgp graphics/povray/povray.bash fgn povray fgh graphics/povray
povray-src(){      echo graphics/povray/povray.bash ; }
povray-source(){   echo ${BASH_SOURCE:-$(env-home)/$(povray-src)} ; }
povray-vi(){       vi $(povray-source) ; }
povray-env(){      elocal- ; }
povray-usage(){ cat << EOU

Povray
=========


* http://www.povray.org/documentation/3.7.0/t2_2.html

* https://github.com/POV-Ray/povray/blob/master/unix/README.md


* http://math.univ-angers.fr/~evain/software/pycao/distributed/documentation/

Pycao is a 3D-modeller for Python. In short, this is a tool to describe a
3D-scene using the Python language. Then you can see your 3D objects using the
Povray raytracer as a plugin.


Current Interest in CSG scene description languages
----------------------------------------------------




EOU
}
povray-dir(){ echo $(local-base)/env/graphics/povray ; }
povray-cd(){  cd $(povray-dir); }
povray-get(){
   local dir=$(dirname $(povray-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d povray ] && git clone https://github.com/POV-Ray/povray

}

povray-doc(){ open http://www.povray.org/documentation/3.7.0/t2_2.html ; }
