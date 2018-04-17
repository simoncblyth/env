# === func-gen- : graphics/unity/unity fgp graphics/unity/unity.bash fgn unity fgh graphics/unity
unity-src(){      echo graphics/unity/unity.bash ; }
unity-source(){   echo ${BASH_SOURCE:-$(env-home)/$(unity-src)} ; }
unity-vi(){       vi $(unity-source) ; }
unity-env(){      elocal- ; }
unity-usage(){ cat << EOU

Unity 5 : free personal edition
=================================

Introductions
--------------

* https://www.raywenderlich.com/142603/unity-tutorial-part-1-getting-started


Fully featured free version
-----------------------------

* http://www.raywenderlich.com/97546/whats-new-unity-5
* http://www.cgchannel.com/2015/03/unity-launches-new-free-unity-5-personal-edition/

All the features of Unity 5, free to anyone earning under $100,000 a year
Unlike the old free version, which included only part of Unity’s feature set,
the new Unity 5 Personal Edition is the full deal. With it, solo artists and
indie studios can publish to any of the 21 platforms Unity now supports.

The only real indication that you’re using a free tool is the splash screens of
games published with the Personal Edition, which carries Unity’s branding.


Physically Based Shading
--------------------------

Unite 2014 - Mastering Physically Based Shading

* https://www.youtube.com/watch?v=eoXb-f_pNag




EOU
}
unity-dir(){ echo $(local-base)/env/graphics/unity/graphics/unity-unity ; }
unity-cd(){  cd $(unity-dir); }
unity-mate(){ mate $(unity-dir) ; }
unity-get(){
   local dir=$(dirname $(unity-dir)) &&  mkdir -p $dir && cd $dir

}
