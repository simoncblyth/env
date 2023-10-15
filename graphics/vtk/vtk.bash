vtk-vi(){       vi $BASH_SOURCE ; }
vtk-env(){      elocal- ; }
vtk-usage(){ cat << EOU

VTK Visualization Toolkit
==========================

Aimed at data visualization rather than more generic 3D graphics.

* http://www.vtk.org/

* http://www.kitware.com/media/html/KitwareOnAppleOSXItJustWorksInMacPorts.html

* https://lorensen.github.io/VTKExamples/site/


installs
----------

::

    (base) epsilon:logs blyth$ conda list vtk
    # packages in environment at /Users/blyth/miniconda3:
    #
    # Name                    Version                   Build  Channel
    vtk                       8.2.0           py37hd5eadda_218    conda-forge


vtk geometry shader
---------------------

* https://discourse.vtk.org/t/geometry-shader-implementation/1189

* https://github.com/thechargedneutron/GSoC-Codes




vtk numpy
------------

* https://blog.kitware.com/improved-vtk-numpy-integration/

* https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/

* https://pyscience.wordpress.com/2014/09/03/ipython-notebook-vtk/


tvtk : used by mayavi
-----------------------

* https://docs.enthought.com/mayavi/tvtk/README.html


Examples
----------

* https://lorensen.github.io/VTKExamples/site/Python/Modelling/Bottle/



pyvista
--------

3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)

* https://docs.pyvista.org/why.html

* see pyvista-


vtk custom shaders
---------------------

* https://vtk.org/Wiki/Shaders_In_VTK

vtkOpenGLShaderCache is the main entrance point for creating and binding shaders.

* https://gitlab.kitware.com/vtk/vtk/-/blob/master/Rendering/OpenGL2/vtkOpenGLShaderCache.cxx
* https://gitlab.kitware.com/vtk/vtk/blob/v7.0.0.rc2/Rendering/OpenGL2/Testing/Cxx/TestUserShader.cxx

::

    vtkNew<vtkOpenGLPolyDataMapper> mapper;
    ...
    mapper->AddShaderReplacement(
        vtkShader::Vertex,
        "//VTK::Normal::Dec", // replace the normal block
        true, // before the standard replacements
        "//VTK::Normal::Dec\n" // we still want the default
        "  varying vec3 myNormalMCVSOutput;\n", //but we add this
        false // only do it once
        );



vtk 9.0.0
------------

* https://blog.kitware.com/vtk-9-0-0-available-for-download/

vtkOpenGLFluidMapper is a new mapper for the real-time rendering of particle-based fluid simulation data.


* https://github.com/pyvista/pyvista/issues/562

  pyvista and VTK9


* https://gitlab.kitware.com/vtk/vtk/-/tree/master/Rendering/OpenGL2



macports
--------

::

    port info vtk   


Looking at the vtk that pyvista is using
------------------------------------------

::

    In [1]: import vtk

    In [2]: vtk.__file__
    Out[2]: '/Users/blyth/miniconda3/lib/python3.7/site-packages/vtkmodules/all.py'

    epsilon:site-packages blyth$ pwd
    /Users/blyth/miniconda3/lib/python3.7/site-packages
    epsilon:site-packages blyth$ cat vtk-8.2.0.egg-info
    Metadata-Version: 2.1
    Name: vtk
    Version: 8.2.0
    Summary: VTK is an open-source toolkit for 3D computer graphics, image processing, and visualization
    Platform: UNKNOWN
    epsilon:site-packages blyth$ 


VTK issue : notes lots to do with OpenXR
-------------------------------------------

* https://gitlab.kitware.com/vtk/vtk/-/issues
* https://examples.vtk.org/site/


VTK C++ Example
-----------------

* https://examples.vtk.org/site/Cxx/GeometricObjects/CylinderExample/


* pyvista VTK on macOS is too old::

   cd ~/env/graphics/vtk/vtk_examples/CylinderExample
   cmake -DVTK_DIR:PATH=$(vtk-dir)/lib/cmake/vtk-8.2
   ## NOPE LOTS OF MISSING MODULES : THAT MUST BE TOO OLD

* https://examples.vtk.org/site/Cxx/IO/ReadPLOT3D/



N install into /data/blyth/vtk/source aka ~/vtk/source
--------------------------------------------------------

* https://docs.vtk.org/en/latest/build_instructions/index.html
* used ccmake to configure with prefix /usr/local/vtk following 
  instructions at above link 

::

   git clone --recursive https://gitlab.kitware.com/vtk/vtk.git ~/vtk/source
   
Installed into /usr/local/vtk/lib64/cmake/vtk-9.3/


VTK Shaders
--------------

* https://gitlab.kitware.com/vtk/vtk/-/tree/master/Rendering/OpenGL2/glsl
* https://gitlab.kitware.com/vtk/vtk/-/blob/master/Rendering/OpenGL2/glsl/readme.txt

* https://examples.vtk.org/site/Cxx/Shaders/BozoShaderDemo/

Has example of updating a uniform in the shader

* https://examples.vtk.org/site/Cxx/Utilities/BrownianPoints/
* https://examples.vtk.org/site/Cxx/VolumeRendering/OpenVRVolume/


VTK Geometry Shader
--------------------

* https://discourse.vtk.org/t/geometry-shader-implementation/1189

how to inject Geometry Shader Code?

In VTK 8: vtkOpenGLPolyDataMapper::AddShaderReplacement 8
In VTK 9 (and current master): vtkShaderProperty::AddGeometryShaderReplacement 20

I found that AddShaderReplacement didn’t work from Python for geometry shaders,
so I have used vtkOpenGLPolyDataMapper::SetGeometryShaderCode 6.

You can see a simple example here 45 that takes in a cube and replaces each
vertex with a pyramid. The example uses the FURY 11 library, but the shader
part is raw VTK.

* https://github.com/dmreagan/fury-shaders/blob/master/geometry.py



VTK USD
---------

* https://discourse.vtk.org/t/universal-scene-description-usd-scene-importer-exporter/6452/3
* https://docs.omniverse.nvidia.com/connect/latest/paraview.html

NVIDIA Omniverse ParaView plugin  



EOU
}
vtk-dir(){ echo $HOME/miniconda3/pkgs/vtk-8.2.0-py37hd5eadda_218 ; }
vtk-cd(){  cd $(vtk-dir); }


vtk-comment(){ cat << EOC
vtk-dir(){ echo $(local-base)/env/graphics/vtk/$(vtk-name); }
vtk-vers(){ echo 6.0.0 ; }
vtk-name(){ echo VTK$(vtk-vers) ; }
vtk-url(){  echo http://www.vtk.org/files/release/6.0/vtk-$(vtk-vers).tar.gz ; }
vtk-get(){
   local dir=$(dirname $(vtk-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(vtk-url)
   local tgz=$(basename $url)
   [ ! -f "$tgz" ] && curl -L -O $url 

   local nam=$(vtk-name)
   [ ! -d "$nam" ] && tar zxvf $tgz
}
EOC
}



