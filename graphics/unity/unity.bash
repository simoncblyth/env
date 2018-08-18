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


Unity Particle System for data visualization
----------------------------------------------

* https://github.com/alchemz/Unity_DataVisualization
* https://github.com/sugi-cho/Unity-GPU-Particle
* https://github.com/antoinefournier/XParticle

* https://github.com/keijiro/KvantSpray

  Object instancing/particle animation system for Unity

* https://github.com/search?q=kvant+user%3Akeijiro&type=Repositories

* https://github.com/keijiro/KvantSwarm
* https://github.com/keijiro/KvantStream
* https://github.com/keijiro/KvantWall


Shaders in Unity
-------------------

* https://unity3d.com/learn/tutorials/topics/graphics/gentle-introduction-shaders


Octane Renderer for Unity Integration
--------------------------------------

* https://unity.otoy.com/guides/installation/


Unity GPU Instancing
---------------------

* https://docs.unity3d.com/Manual/GPUInstancing.html


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
