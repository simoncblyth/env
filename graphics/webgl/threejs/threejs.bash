# === func-gen- : graphics/webgl/threejs/threejs fgp graphics/webgl/threejs/threejs.bash fgn threejs fgh graphics/webgl/threejs
threejs-src(){      echo graphics/webgl/threejs/threejs.bash ; }
threejs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(threejs-src)} ; }
threejs-vi(){       vi $(threejs-source) ; }
threejs-env(){      elocal- ; }
threejs-usage(){ cat << EOU

THREEJS
========

Javascript 3D library built ontop of WebGL/Canvas/SVG/CSS3D

* http://threejs.org/
* https://github.com/mrdoob/three.js/
* https://github.com/mrdoob/three.js/wiki

EOU
}

threejs-name(){ echo three.js ; }
threejs-dir(){ echo $(local-base)/env/graphics/webgl/$(threejs-name) ; }  # yes thats a funny dir name
threejs-cd(){  cd $(threejs-dir); }
threejs-mate(){ mate $(threejs-dir) ; }
threejs-get(){
   local dir=$(dirname $(threejs-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(threejs-name)
   [ ! -d "$nam" ] && git clone https://github.com/mrdoob/three.js.git 
}

threejs-minified-url(){ echo http://threejs.org/build/three.min.js ; }
threejs-minified-get(){ 
    local dir=$(dirname $(threejs-dir)) &&  mkdir -p $dir && cd $dir
    curl -L -O $(threejs-minified-url) ; 
}

threejs-examples(){
    cd $(env-home)/graphics/webgl/threejs/examples
    python -m SimpleHTTPServer
    open http://localhost:8000
}



