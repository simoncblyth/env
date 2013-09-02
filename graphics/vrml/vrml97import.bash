# === func-gen- : graphics/vrml/vrml97import fgp graphics/vrml/vrml97import.bash fgn vrml97import fgh graphics/vrml
vrml97import-src(){      echo graphics/vrml/vrml97import.bash ; }
vrml97import-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vrml97import-src)} ; }
vrml97import-vi(){       vi $(vrml97import-source) ; }
vrml97import-env(){      elocal- ; }
vrml97import-usage(){ cat << EOU

VRML97 IMPORT PLUGIN FOR BLENDER
==================================

* http://vrml97import.sourceforge.net/

BLENDER 2.59 INCORPORATES VRML97 IMPORT AS STANDARD

* unclear how/if that standard import plugin differs from this old optional plugin


EOU
}
vrml97import-dir(){ echo $(local-base)/env/graphics/vrml/graphics/vrml-vrml97import ; }
vrml97import-cd(){  cd $(vrml97import-dir); }
vrml97import-mate(){ mate $(vrml97import-dir) ; }
vrml97import-get(){
   local dir=$(dirname $(vrml97import-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://downloads.sourceforge.net/project/vrml97import/vrml97import/VRML97_Import_v0.33/vrml97_import_v0.33.zip
   local zip=$(basename $url)
   [ ! -f "$zip" ] && curl -L -O $url

}
