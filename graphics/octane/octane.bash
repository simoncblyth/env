# === func-gen- : graphics/octane/octane fgp graphics/octane/octane.bash fgn octane fgh graphics/octane
octane-src(){      echo graphics/octane/octane.bash ; }
octane-source(){   echo ${BASH_SOURCE:-$(env-home)/$(octane-src)} ; }
octane-vi(){       vi $(octane-source) ; }
octane-env(){      elocal- ; }
octane-usage(){ cat << EOU

Octane : Unbiased GPU ray tracer from OTOY
===============================================



* https://blog.paperspace.com/gpu-rendering-with-octane-in-the-cloud/

Octane is the world's first and fastest GPU-accelerated, unbiased, physically
correct renderer. Octane was developed by a New Zealand-based company called
Refractive Software, Ltd, and was later taken over by the Company OTOY in 2012.
Now it is being used in feature films, commercials, and even architectural
rendering. With the ability to run Octane in the cloud, it is likely to become
even more integrated into the creative process of these industries.



* https://en.m.wikipedia.org/wiki/Octane_Render

It is the first commercially available unbiased renderer to work exclusively on the GPU 


OTOY : CUDA on AMD/ARM 
------------------------

* https://venturebeat.com/2018/03/19/otoys-octane-4-blends-cinematic-video-and-games-using-ai/
* https://venturebeat.com/2016/03/09/otoy-breakthrough-lets-game-developers-run-the-best-graphics-software-across-platforms/




ORBX 3D file format
---------------------

* https://home.otoy.com/tag/orbx/

Plans for open source: OTOY intends to open source the ORBX media format. It
will be available on GitHub when OctaneRender 3 is made available. The format
works in JavaScript and LUA – no native code is required.



OTOY/Unity tie up 
--------------------

* https://uploadvr.com/otoy-building-one-vrs-important-technologies/

(November 2016)

At Unity 3D’s conference in L.A. last week, OTOY announced their technology
would be natively integrated into the Unity engine. Since Unity is the world’s
most popular game engine with around 45 percent market share, this graduates
ORBX from a power-user niche to a widely accessible creation format. When
OTOY’s integration releases next year, you’ll be able to import and export ORBX
directly from Unity, and use OTOY’s Octane Renderer to create experiences with
extremely realistic visual detail. At launch, this means you’ll be able create
and edit photorealistic scenes directly in Unity, then export a 360 video in
the streamable ORBX format. However, it won’t be long before you can move
through a scene as positionally tracked lightfields become supported (more on
that later).



Imagination/PowerVR dedicated ray tracing chips 
--------------------------------------------------

* :google:`imagination wizard gpu`

PowerVR Furian Architecture (March 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.anandtech.com/show/11186/imagination-announces-powervr-furian-gpu-architecture

Octane/PowerVR 
~~~~~~~~~~~~~~~~~~~~

* https://home.otoy.com/otoy-and-imagination-unveil-breakthrough-powervr-ray-tracing-platform/


OctaneRender 4 prototype achieves 100+ million rays/second on a 2 watt PowerVR
Ray Tracing mobile GPU core – a 10x increase in ray-tracing performance/watt
compared to GPGPU compute ray tracing in OctaneRender 3”

OctaneRender 4, running on the current PowerVR Wizard GR6500 Ray Tracing
hardware, is able to trace over 100 million rays per second in fully dynamic
scenes, within a power envelope suitable for lightweight mobile, VR and AR
glasses. According to Imagination, these results are just the tip of the
iceberg. The company expects performance to significantly increase based on
improvements already in the pipeline.


Imagination Wizard GPU API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://imagination-technologies-cloudfront-assets.s3.amazonaws.com/idc-docs/gdc16/2_Ray%20Tracing%20API%20Overview.pdf




EOU
}
octane-dir(){ echo $(local-base)/env/graphics/octane/graphics/octane-octane ; }
octane-cd(){  cd $(octane-dir); }
octane-mate(){ mate $(octane-dir) ; }
octane-get(){
   local dir=$(dirname $(octane-dir)) &&  mkdir -p $dir && cd $dir

}
