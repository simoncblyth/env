# === func-gen- : graphics/vrml/h3d fgp graphics/vrml/h3d.bash fgn h3d fgh graphics/vrml
h3d-src(){      echo graphics/vrml/h3d.bash ; }
h3d-source(){   echo ${BASH_SOURCE:-$(env-home)/$(h3d-src)} ; }
h3d-vi(){       vi $(h3d-source) ; }
h3d-env(){      elocal- ; }
h3d-usage(){ cat << EOU

H3D
===

H3D Introduction
-----------------

* http://www.h3dapi.org/

H3DAPI is an open source haptics software development platform that uses the
open standards OpenGL and X3D with haptics in one unified scene graph to handle
both graphics and haptics. H3DAPI is cross platform and haptic device
independent. It enables audio integration as well as stereography on supported
displays.

Unlike most other scene graph APIs, H3DAPI is designed chiefly to support a
special rapid development process. By combining X3D, C++ and the scripting
language Python, H3DAPI offers three ways of programming applications that
offer the best of both worlds ? execution speed where performance is critical,
and development speed where performance is less critical.

H3DAPI is written in C++, and is designed to be extensible, ensuring that
developers possess the freedom and means to customize and add any needed
haptics or graphics features in H3DAPI for their applications.

H3DAPI has been used to develop a diverse range of haptics and multimodal
applications in various fields including but not limited to dental, medical,
industrial and visualization. To encourage learning and growth in the use of
haptics technology, H3DAPI is open source and released under the GNU GPL
license, with options for commercial licensing.


H3D Documentation
-------------------

* http://www.h3dapi.org/uploads/api/H3DAPI_2.2/doc/H3D%20API%20Manual.pdf






EOU
}
h3d-dir(){ echo $(local-base)/env/graphics/vrml/graphics/vrml-h3d ; }
h3d-cd(){  cd $(h3d-dir); }
h3d-mate(){ mate $(h3d-dir) ; }
h3d-get(){
   local dir=$(dirname $(h3d-dir)) &&  mkdir -p $dir && cd $dir

}
