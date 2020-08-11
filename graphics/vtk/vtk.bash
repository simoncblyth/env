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



