# === func-gen- : graphics/openscenegraph/osgdata fgp graphics/openscenegraph/osgdata.bash fgn osgdata fgh graphics/openscenegraph
osgdata-src(){      echo graphics/openscenegraph/osgdata.bash ; }
osgdata-source(){   echo ${BASH_SOURCE:-$(env-home)/$(osgdata-src)} ; }
osgdata-vi(){       vi $(osgdata-source) ; }
osgdata-env(){      elocal- ; }
osgdata-usage(){ cat << EOU

OpenSceneGraph Data 
======================



EOU
}
osgdata-name(){ echo OpenSceneGraph-Data-3.0.0 ; }
osgdata-dir(){ echo $(local-base)/env/graphics/openscenegraph/$(osgdata-name) ; }
osgdata-cd(){  cd $(osgdata-dir); }
osgdata-mate(){ mate $(osgdata-dir) ; }
osgdata-get(){
   local dir=$(dirname $(osgdata-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://www.openscenegraph.org/downloads/stable_releases/OpenSceneGraph-3.0/data/$(osgdata-name).zip
   local zip=$(basename $url)
   local nam=${zip/.zip}
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip 
}
