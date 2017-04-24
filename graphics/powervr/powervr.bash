# === func-gen- : graphics/powervr/powervr fgp graphics/powervr/powervr.bash fgn powervr fgh graphics/powervr
powervr-src(){      echo graphics/powervr/powervr.bash ; }
powervr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(powervr-src)} ; }
powervr-vi(){       vi $(powervr-source) ; }
powervr-env(){      elocal- ; }
powervr-usage(){ cat << EOU

Imagination PowerVR Chips
=============================


* :google:`PowerVR Wizard Ray Tracing GPU`


PowerVR Ray Tracing 
--------------------

* https://www.imgtec.com/powervr/ray-tracing/

Revolutionary technology that brings lightning fast ray tracing 
to the world’s leading mobile GPU. This game-changing feature
enables astonishing realism as well as allowing developers and content creators
to simplify their workflow.

PowerVR Wizard GPUs 
--------------------

The PowerVR Wizard family of GPUs delivers a highly
optimized and ultra-efficient implementation of the PowerVR Ray Tracing
technology. PowerVR Wizard graphics IP processors enable more immersive games
and apps with real-life dynamic lighting models that produce advanced lighting
effects, dynamic soft shadows, and life-like reflections and transparencies,
previously unachievable in a mobile form factor.

PowerVR Wizard Ray Tracing GPU IP processors are highly scalable, making them
disruptive to many markets from mobile to high-end.




EOU
}
powervr-dir(){ echo $(local-base)/env/graphics/powervr/graphics/powervr-powervr ; }
powervr-cd(){  cd $(powervr-dir); }
powervr-mate(){ mate $(powervr-dir) ; }
powervr-get(){
   local dir=$(dirname $(powervr-dir)) &&  mkdir -p $dir && cd $dir

}
