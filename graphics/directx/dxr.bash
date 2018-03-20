# === func-gen- : graphics/directx/dxr fgp graphics/directx/dxr.bash fgn dxr fgh graphics/directx
dxr-src(){      echo graphics/directx/dxr.bash ; }
dxr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dxr-src)} ; }
dxr-vi(){       vi $(dxr-source) ; }
dxr-env(){      elocal- ; }
dxr-usage(){ cat << EOU

DXR : DirectX Ray Tracing (March 2018)
============================================


* https://blogs.msdn.microsoft.com/directx/2018/03/19/announcing-microsoft-directx-raytracing/


NVIDIA pushing Volta, or is it really needed for RTX ?
--------------------------------------------------------

* https://devblogs.nvidia.com/introduction-nvidia-rtx-directx-raytracing/

So will ray tracing always remain a dream of the future, and never arrive in
the present? At GDC 2018, NVIDIA unveiled RTX, a high-performance
implementation that will power all ray tracing APIs supported by NVIDIA on
Volta and future GPUs. At the same event, Microsoft announced the integration
of ray tracing as a first-class citizen into their industry standard DirectX
API.

Putting these two technologies together forms such a powerful combination that
we can confidently answer the above question: the future is here! This is not a
hyperbole: leading game studios are developing upcoming titles using RTX
through DirectX — today. Ray tracing in games is no longer a pipe dream. It’s
happening, and it will usher in a new era of real-time graphics.

* https://developer.nvidia.com/gameworks-ray-tracing


NVIDIA RTX hardware implementation (from Volta) exposed by DXR
----------------------------------------------------------------

* https://www.anandtech.com/show/12546/nvidia-unveils-rtx-technology-real-time-ray-tracing-acceleration-for-volta-gpus-and-later

In conjunction with Microsoft’s new DirectX Raytracing (DXR) API announcement,
today NVIDIA is unveiling their RTX technology, providing ray tracing
acceleration for Volta and later GPUs. Intended to enable real-time ray tracing
for games and other applications, RTX is essentially NVIDIA's DXR backend
implementation. For this NVIDIA is utilizing a mix of software and hardware –
including new microarchitectural features – though the company is not
disclosing further details. Alongside RTX, NVIDIA is also announcing their new
GameWorks ray tracing tools, currently in early access to select development
partners.

With NVIDIA working with Microsoft, RTX is fully supported by DXR, meaning that
all RTX functionality is exposed through the API. And while only Volta and
newer architectures have the specific hardware features required for hardware
acceleration of DXR/RTX, DXR's compatibility mode means that a DirectCompute
path will be available for non-Volta hardware. Beyond Microsoft, a number of
developers and game engines are supporting RTX, with DXR and RTX tech demos at
GDC 2018.

On that note, since the entire “GPU – RTX – DXR – GameWorks Ray Tracing” stack
only applies to Volta, the broader public is essentially limited to the Titan
V, and NVIDIA likewise noted that RTX technology of present was primarily
intended for developer use. For possible ray tracing acceleration on pre-Volta
architectures, NVIDIA only referred back to DXR, though Microsoft has equally
referred back to vendors for hardware-related technical details. And while
strict performance numbers aren’t being disclosed, NVIDIA stated that real time
ray tracing with RTX on Volta would be “integer multiples faster” than with DXR
on older hardware.



DXR described
-----------------

* https://arstechnica.com/gadgets/2018/03/microsoft-announces-the-next-step-in-gaming-graphics-directx-raytracing/

At GDC, Microsoft announced a new feature for DirectX 12: DirectX Raytracing
(DXR). The new API offers hardware-accelerated raytracing to DirectX
applications, ushering in a new era of games with more realistic lighting,
shadows, and materials.

...

The company says that it has been working on DXR for close to a year, and
Nvidia in particular has plenty to say about the matter. Nvidia has its own
raytracing engine designed for its Volta architecture (though currently, the
only video card shipping with Volta is the Titan V, so the application of this
is likely limited). When run on a Volta system, DXR applications will
automatically use that engine.


* https://www.anandtech.com/show/12547/expanding-directx-12-microsoft-announces-directx-raytracing

This morning at GDC 2018 as part of a coordinated release with some of their
hardware and software partners, Microsoft is announcing a major new feature
addition to the DirectX 12 graphics API: DirectX Raytracing. Exactly what the
name says on the tin, DirectX Raytracing will provide a standard API for
hardware and software accelerated ray tracing under DirectX, allowing
developers to tap into the rendering model for newer and more accurate graphics
and effects.

Going hand-in-hand with both new and existing hardware, the DXR command set is
meant to provide a standardized means for developers to implement ray tracing
in a GPU-friendly manner. Furthermore as an extension of the existing DirectX
12 feature set, DXR is meant to be tightly integrated with traditional
rasterization, allowing developers to mix the two rendering techniques to suit
their needs and to use the rendering technique that delivers the best
effects/best performance as necessary.

But like Microsoft’s other DirectX APIs it’s important to note that the company
isn’t defining how the hardware should work, only that the hardware needs to
support certain features. Past that, it’s up to the individual hardware vendors
to create their own backends for executing DXR commands. As a result – and
especially as this is so early – everyone from Microsoft to hardware vendors
are being intentionally vague about how hardware acceleration is going to work.

At the base level, DXR will have a full fallback layer for working on existing
DirectX 12 hardware. As Microsoft’s announcement is aimed at software
developers, they’re pitching the fallback layer as a way for developers to get
started today on using DXR. It’s not the fastest option, but it lets developers
immediately try out the API and begin writing software to take advantage of it
while everyone waits for newer hardware to become more prevalent. 

================   ============================
Vendor              Support
================   ============================
NVIDIA Volta        Hardware + Software (RTX)
NVIDIA Pre-Volta    Software
================   ============================

For today’s reveal, NVIDIA is simultaneously announcing that they will support
hardware acceleration of DXR through their new RTX Technology. RTX in turn
combines previously-unannounced Volta architecture ray tracing features with
optimized software routines to provide a complete DXR backend, while pre-Volta
cards will use the DXR shader-based fallback option. Meanwhile AMD has also
announced that they’re collaborating with Microsoft and that they’ll be
releasing a driver in the near future that supports DXR acceleration. The tone
of AMD’s announcement makes me think that they will have very limited hardware
acceleration relative to NVIDIA, but we’ll have to wait and see just what AMD
unveils once their drivers are available.

Meanwhile DXR will introduce multiple new shader types to handle ray
processing, including ray-generation, closest-hit, any-hit, and miss shaders.
Finally, the 3D world itself will be described using what Microsoft is terming
the acceleration structure, which is a full 3D environment that has been
optimized for GPU traversal.

* http://schedule.gdconf.com/session/directx-evolving-microsofts-graphics-platform-presented-by-microsoft/856594









EOU
}
dxr-dir(){ echo $(local-base)/env/graphics/directx/graphics/directx-dxr ; }
dxr-cd(){  cd $(dxr-dir); }
dxr-mate(){ mate $(dxr-dir) ; }
dxr-get(){
   local dir=$(dirname $(dxr-dir)) &&  mkdir -p $dir && cd $dir

}
