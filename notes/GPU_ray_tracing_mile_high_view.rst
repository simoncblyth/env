notes/GPU_ray_tracing_mile_high_view.rst 
============================================

NVIDIA
-------


NVIDIA : OptiX : A Ray Tracing Framework in CUDA
--------------------------------------------------

* https://developer.nvidia.com/rtx/ray-tracing/optix


NVIDIA : Vulkan
-----------------

* https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/



AMD
----

* :google:`AMD ray tracing`
* https://www.pcgamesn.com/amd/new-ray-tracing-rdna-4

we’ve found that the ray tracing performance of current AMD RDNA 3 GPUs is
roughly comparable to that of Nvidia’s last-gen Ampere graphics cards. RDNA 3
definitely improved on RDNA 2 when it comes to ray tracing, for example by
moving some of the hardware bounding volume hierarchy (BVH) sorting and
traversal work from the GPU shaders to the RT cores, but Nvidia’s Ada GPUs are
still much quicker at ray tracing. 


* https://gpuopen.com/radeon-raytracing-analyzer/


* https://www.howtogeek.com/its-2024-can-amd-graphics-cards-handle-ray-tracing-yet/


AMD : HIPRT: A Ray Tracing Framework in HIP
-------------------------------------------------

* https://gpuopen.com/download/publications/HIPRT-paper.pdf
* ~/opticks_refs/HIPRT-paper.pdf

There are two major GPU-based industrial ray tracing frameworks oriented on
professional rendering: Embree [Wald et al . 2014] for Intel GPUs and OptiX
[Parker et al . 2010] for Nvidia GPUs. We introduce HIPRT, a ray tracing
framework written in the HIP1 kernel language, tailored for professional
rendering on AMD GPUs.


* https://github.com/GPUOpen-LibrariesAndSDKs/HIPRTSDK/blob/main/tutorials/readme.md
* https://gpuopen.com/hiprt/
* https://radeon-pro.github.io/RadeonProRenderDocs/en/hiprt/about.html


HIP RT is a ray tracing library for HIP, making it easy to write ray-tracing
applications in HIP. The APIs and library are designed to be minimal, lower
level, and simple to use and integrate into any existing HIP applications.

Although there are other ray tracing APIs which introduce many new things, we
designed HIP RT in a slightly different way so you do not need to learn many
new kernel types.


* https://gpuopen.com/radeon-rays/

* https://gpuopen.com/download/publications/HPLOC.pdf
* ~/opticks_refs/HPLOC.pdf



Intel
-----

* :google:`intel Arc Alchemist GPUs`
* :google:`intel Xe`
* https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/iris-xe-integrated-graphics/overview.html

* https://wccftech.com/intel-xe2-gpus-50-percent-uplift-new-ray-tracing-cores-lunar-lake-arc-battlemage-discrete/


A major block of the Xe2 core is its RTU (Ray Tracing Unit) which features 3
traversal pipelines, 18 box intersections (6 per Box intersection & 3 boxes per
RTU), and 2 triangle intersections.


Intel : Embree on Intel ARC GPUs
---------------------------------

* https://www.embree.org/

Intel® Embree is a high-performance ray tracing library developed at Intel
which supports x86 CPUs under Linux, macOS, and Windows; ARM CPUs on macOS; as
well as Intel® Arc™ GPUs under Linux and Windows. Embree targets graphics
application developers to improve the performance of photo-realistic rendering
applications and is optimized towards production rendering. Embree is released
as open source under the Apache 2.0 License . 

Embree supports hardware accelerated ray tracing on Intel GPUs through the SYCL
programming language for excellent GPU rendering performance. 



Apple
-----

* :google:`apple metal ray tracing`
* https://developer.apple.com/documentation/metal/ray_tracing_with_acceleration_structures



Qualcomm
---------

* :google:`Qualcomm ray tracing`


Huawei
-------

* :google:`Huawei ray tracing`
* https://consumer.huawei.com/uk/community/details/Huawei-Pheonix-will-bring-ray-tracing-tech-to-smartphone-gaming/topicId_41723/

* :google:`huawei Phoenix Engine`


* https://www.huaweicentral.com/huawei-brings-harmonyos-next-cloud-rendering-for-realistic-gaming-experience/

Starting with the cloud rendering technology, Huawei says that HarmonyOS NEXT
will enhance the gaming experience.

Cloud rendering helps in achieving real-time vibrant renders of over 1 billion
rays per second.

Next capability of the cloud rendering technology is PC-level ray tracing. From
the name, the feature uses light to add more realism to video games. It works
on how the light reflects or refracts on a particular object and hereafter,
renders the image.

These capabilities together enhance the gaming quality and overall experience.
Apart from the games, the company has integrated more facilities into HarmonyOS
NEXT which makes other applications more appropriate for Huawei devices.


* https://www.gsmarena.com/huaweis_harmonyos_next_beta_launches_officially-news-63397.php


23 June 2024

Huawei Mobile software HarmonyOS

Yes, it is finally happening. Huawei's HarmonyOS NEXT is shedding its janky
Android roots. As promised at the Huawei Developer Conference (HDC) 2023, the
OS's new version is a new and fresh software effort. Huawei officially launched
the HarmonyOS NEXT Beta at this year's HDC 2024 conference.

Let's go through some of the basics first. HarmonyOS NEXT is not based on
Android, nor is it strictly speaking a Linux system. It is based on OpenHarmony
and is a microkernel-based core distributed operating system for HarmonyOS.
That is to say that HarmonyOS NEXT has the HarmonyOS microkernel. It only
supports native APP apps via Ark Compiler with Huawei Mobile Services (HMS)
Core support. That means that the OS has no native compatibility with Android
APK apps and files, as demonstrated in this post originally from Weibo.

...

HarmonyOS NEXT also promises to be a multimedia powerhouse with support for Ark
Multimedia Engine, Ark Graphics Engine, and FFRT. That last tech apparently
brings the promise of PC-level ray tracing for games.


* https://developer.huawei.com/consumer/en/doc/harmonyos-guides-V5/ffrt-kit-V5

* Function Flow Runtime Kit 

The Function Flow programming model allows you to develop an application by
creating tasks and describing their dependencies. Its most outstanding features
are task-based and data-driven.



* :google:`harmonyos development on workstation`

HarmonyOS Next hasn't yet arrived on PC, but recent leaks suggest it soon will,
paving the way for a new Chinese homegrown desktop OS. Images of HarmonyOS Next
for PC suggest an operating system taking design cues from MacOS. The system
has a familiar status bar and dock bar combo on the top and bottom.Jul 17, 2024


* https://www.tomshardware.com/software/operating-systems/huawei-long-awaited-windows-challenger-will-likely-come-to-pcs-this-year

According to industry analysts, Huawei is expected to release a PC version of
its HarmonyOS Next operating system before the end of the year. Huawei's
developer website is seeing a growth in images featuring HarmonyOS running on
PC, as spotted by X (formerly Twitter) user and HarmonyOS developer
jasonwill101 on X.

HarmonyOS is Huawei's operating system for its phones and tablets. It was
created in 2019 after heavy U.S. sanctions forced Huawei out of the United
States and blocked its access to the Android operating system. HarmonyOS is
based heavily on the open-source version of Android (AOSP) but was far enough
away for Huawei to be able to continue producing its high-end smartphones.

HarmonyOS Next, on the other hand, is an Android-free variant of HarmonyOS. The
new operating system doesn't use AOSP libraries, can't run .apk files, and is a
significant step towards complete independence from US-based software for the
vendor. HarmonyOS Next is not currently shipping with Huawei products but is
available as a developer sandbox to develop and test apps for native HarmonyOS
use. HarmonyOS Next hasn't yet arrived on PC, but recent leaks 
suggest it soon will, paving the way for a new Chinese homegrown desktop OS.

Images of HarmonyOS Next for PC suggest an operating system taking design cues
from MacOS. The system has a familiar status bar and dock bar combo on the top
and bottom. The fullscreen/minimize/close buttons live on the right-hand side
of programs, mirroring MacOS's traffic light system.

Huawei's recent strategy for HarmonyOS has publicly been phones-first. With
HarmonyOS being open-source, much like Android, widespread adoption across the
Chinese market and beyond outside of Huawei phones is possible and a big goal
for Huawei. HarmonyOS already makes up 16% of the Chinese phone market, which
is expected to grow in the coming years.

While Huawei may want to focus development efforts on HarmonyOS towards phones,
Chinese governments, local and national, have other plans. The regional
government of Shenzhen, the metropolis that links Hong Kong to the Chinese
mainland, recently began the 'Shenzhen Action Plan for Supporting the
Development of Native HarmonyOS Open Source Applications in 2024.' The action
plan includes ways Shenzhen seeks to boost HarmonyOS adoption and development,
with a significant goal of Shenzhen accounting for 10% of the HarmonyOS
products in China by the end of 2024.



* :google:`Shenzhen Action Plan for Supporting the Development of Native HarmonyOS Open Source Applications in 2024.`



* https://sa2021.siggraph.org/en/attend/exhibitor-talks/21/session_slot/641

  Huawei real-time mobile ray tracing for flagship phones and games



HarmonyOS/Hongmeng
---------------------

* https://www.sz.gov.cn/en_szgov/news/latest/content/post_11202141.html

HarmonyOS, or Hongmeng in Chinese, is an open-source operating system designed
for various devices including intelligent screens, tablets, wearables, and
cars. It was first launched in August 2019.


HarmonyOS, as a next-generation operating system for smart devices, provides a
common language that allows different kinds of devices to connect and
communicate and gives users a more convenient, seamless, and secure experience,
according to Huawei.


The operating system has been released at a time when the United States
continues to bar Huawei from accessing key American technologies and products
in an attempt to lock the company out of the global 5G market. 


* https://www.techpowerup.com/forums/threads/china-pushes-adoption-of-huaweis-harmonyos-to-replace-windows-ios-and-android.321296/


Huawei Cloud
------------

* https://www.huaweicloud.com/intl/en-us/product/gpu.html
* https://support.huaweicloud.com/intl/en-us/productdesc-ecs/ecs_01_0045.html


Vulkan
--------

* https://people.ece.ubc.ca/~aamodt/publications/papers/saed.micro2022.pdf
* ~/opticks_refs/VulkanSim_saed_micro2022.pdf

  Vulkan-Sim: A GPU Architecture Simulator for Ray Tracing



Cross vendor projects
-----------------------

* https://github.com/RayTracing/gpu-tracing
* https://github.com/gfx-rs/wgpu

wgpu is a cross-platform, safe, pure-rust graphics API. It runs natively on
Vulkan, Metal, D3D12, and OpenGL; and on top of WebGL2 and WebGPU on wasm.

* https://gpuweb.github.io/gpuweb/

* :google:`WebGPU ray tracing`
* https://github.com/codedhead/webrtx
* https://github.com/maierfelix/dawn-ray-tracing/tree/master


* https://wiki.archlinux.org/title/Hardware_raytracing


