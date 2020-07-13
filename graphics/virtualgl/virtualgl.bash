# === func-gen- : graphics/virtualgl/virtualgl fgp graphics/virtualgl/virtualgl.bash fgn virtualgl fgh graphics/virtualgl
virtualgl-src(){      echo graphics/virtualgl/virtualgl.bash ; }
virtualgl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(virtualgl-src)} ; }
virtualgl-vi(){       vi $(virtualgl-source) ; }
virtualgl-env(){      elocal- ; }
virtualgl-usage(){ cat << EOU

VirtualGL
===========

* http://www.virtualgl.org
* http://sourceforge.net/projects/virtualgl/

* https://github.com/VirtualGL/virtualgl

* http://www.virtualgl.org/vgldoc/2_2_1/



VirtualGL redirects 3D commands from a Unix/Linux OpenGL application onto a
server-side GPU and converts the rendered 3D images into a video stream with
which remote clients can interact to view and control the 3D application in
real time.


See Also
------------

* vgpu-


EGL Background
---------------------

* https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
* https://developer.nvidia.com/blog/interactive-supercomputing-in-situ-visualization-tesla-gpus/





Access the GPU without going through an X server 
---------------------------------------------------

* https://github.com/VirtualGL/virtualgl/issues/10

* https://gist.github.com/dcommander/ee1247362201552b2532



Remote Visualization
----------------------

* http://www.nice-software.com/products/dcv

* http://insidehpc.com/2014/11/nice-desktop-cloud-visualization-dcv-2014-allows-unprecedented-cadcae-visualization-experience-cloud/






EOU
}
virtualgl-dir(){ echo $(local-base)/env/graphics/virtualgl/graphics/virtualgl-virtualgl ; }
virtualgl-cd(){  cd $(virtualgl-dir); }
virtualgl-mate(){ mate $(virtualgl-dir) ; }
virtualgl-get(){
   local dir=$(dirname $(virtualgl-dir)) &&  mkdir -p $dir && cd $dir

}
