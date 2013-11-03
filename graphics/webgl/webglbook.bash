# === func-gen- : graphics/webgl/webglbook fgp graphics/webgl/webglbook.bash fgn webglbook fgh graphics/webgl
webglbook-src(){      echo graphics/webgl/webglbook.bash ; }
webglbook-source(){   echo ${BASH_SOURCE:-$(env-home)/$(webglbook-src)} ; }
webglbook-vi(){       vi $(webglbook-source) ; }
webglbook-env(){      elocal- ; }
webglbook-usage(){ cat << EOU

WEBGL BOOK
===========

Code Examples for Tony Parisi's Book, WebGL Up and Running

* https://github.com/tparisi/WebGLBook

Examples
---------

::

    simon:Chapter 1 blyth$ pwd
    /usr/local/env/graphics/webgl/WebGLBook/Chapter 1
    simon:Chapter 1 blyth$ vi example1-1.html 
    simon:Chapter 1 blyth$ python -m SimpleHTTPServer
    Serving HTTP on 0.0.0.0 port 8000 ...

    http://localhost:8000/
    

Chapter 2
~~~~~~~~~~~

* example2-2.html  nothing showing with SVG or Canvas renderer



Fallback to Canvas Renderer
----------------------------

* http://stackoverflow.com/questions/9899807/three-js-detect-webgl-support-and-fallback-to-regular-canvas
* https://github.com/mrdoob/three.js/blob/master/examples/js/Detector.js

::

   renderer = Detector.webgl? new THREE.WebGLRenderer(): new THREE.CanvasRenderer();


FUNCTIONS
----------

*webglbook-srv*
       run python SimpleHTTPServer from WebGLBook directory. 
       Does not work from subdirectories as presumably "../" 
       traversal is disabled for security reasons. 



EOU
}
webglbook-dir(){ echo $(local-base)/env/graphics/webgl/WebGLBook ; }
webglbook-cd(){  cd "$(webglbook-dir)/$1" ; }
webglbook-ch(){  webglbook-cd Chapter\ ${1:-7} ; }
webglbook-mate(){ mate $(webglbook-dir) ; }
webglbook-get(){
   local dir=$(dirname $(webglbook-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(basename $(webglbook-dir))
   [ ! -d "$nam" ] && git clone  https://github.com/tparisi/WebGLBook
}
webglbook-srv(){
   webglbook-cd
   python-
   python-srv
   open http://localhost:8000
}
webglbook-diff(){
   webglbook-cd
   git status
   git diff
}
