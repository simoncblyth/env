# === func-gen- : graphics/renderer/renderer fgp graphics/renderer/renderer.bash fgn renderer fgh graphics/renderer
renderer-src(){      echo graphics/renderer/renderer.bash ; }
renderer-source(){   echo ${BASH_SOURCE:-$(env-home)/$(renderer-src)} ; }
renderer-vi(){       vi $(renderer-source) ; }
renderer-env(){      elocal- ; }
renderer-usage(){ cat << EOU

Lists of Renderers
=====================




Open Source Ray Tracers with CSG ?
-------------------------------------

BRL-CAD
~~~~~~~~~~~

* https://brlcad.org
* US Military, suspect scene authoring via GUI, not separate language

Povray
~~~~~~~~

* see povray-  


Commercial 
--------------

* http://joomla.renderwiki.com/joomla/index.php?option=com_content&view=article&id=84&Itemid=75

Free
-------

* http://joomla.renderwiki.com/joomla/index.php?option=com_content&view=article&id=86&Itemid=76


EOU
}
renderer-dir(){ echo $(local-base)/env/graphics/renderer/graphics/renderer-renderer ; }
renderer-cd(){  cd $(renderer-dir); }
renderer-mate(){ mate $(renderer-dir) ; }
renderer-get(){
   local dir=$(dirname $(renderer-dir)) &&  mkdir -p $dir && cd $dir

}
