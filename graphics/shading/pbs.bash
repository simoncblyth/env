# === func-gen- : graphics/shading/pbs fgp graphics/shading/pbs.bash fgn pbs fgh graphics/shading
pbs-src(){      echo graphics/shading/pbs.bash ; }
pbs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pbs-src)} ; }
pbs-vi(){       vi $(pbs-source) ; }
pbs-env(){      elocal- ; }
pbs-usage(){ cat << EOU

PBS : Physically Based Shading in Theory and Practice
========================================================

* http://blog.selfshadow.com
* http://blog.selfshadow.com/publications/s2012-shading-course/#course_content


SIGGRAPH 2014 Course 
---------------------

* http://blog.selfshadow.com/publications/s2014-shading-course/


Background: Physics and Math of Shading (Naty Hoffman)
--------------------------------------------------------

* http://blog.selfshadow.com/publications/s2014-shading-course/hoffman/s2014_pbs_physics_math_slides.pdf

  * Fresnel reflectance curves as a function of incident angle (0:90)
    for variety of materials 

    * metals stay high ~0.9, dielectrics have drastic increase from above 60 degrees to near 1 

  * microfacet surface models, various formulations


Physically Based Shader Design in Arnold (Anders Langlands) 
-------------------------------------------------------------

* http://blog.selfshadow.com/publications/s2014-shading-course/langlands/s2014_pbs_alshaders_slides.pdf
* http://blog.selfshadow.com/publications/s2014-shading-course/langlands/s2014_pbs_alshaders_notes.pdf
* https://bitbucket.org/anderslanglands/alshaders/wiki/Home

* notably realistic glass renders


Physically Based Shading at Disney (Brent Burley) 
--------------------------------------------------

* http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_slides_v2.pdf
* http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
* http://www.disneyanimation.com/technology/brdf.html


BRDF Tools
~~~~~~~~~~~~

* https://github.com/wdas/brdf
* http://brdflab.sourceforge.net


BRDF : bidirectional reflectance distribution function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.merl.com/brdf/
* http://people.csail.mit.edu/wojciech/BRDFDatabase/
* http://people.csail.mit.edu/wojciech/BRDFDatabase/code/BRDFRead.cpp

  * measured reflectance functions of 100 different materials

Provides RGB color from material BRDF table and 4 parameters: theta/phi 
for incident + reflected directions::

    // Given a pair of incoming/outgoing angles, look up the BRDF.
    void lookup_brdf_val(double* brdf, double theta_in, double fi_in,
                  double theta_out, double fi_out, 
                  double& red_val,double& green_val,double& blue_val)


BRDF wavelength dependent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`brdf wavelength dependent`

* http://www.cs.ucla.edu/~zhu/tutorial/An_Introduction_to_BRDF-Based_Lighting.pdf
* http://cg.informatik.uni-freiburg.de/course_notes/graphics2_07_materials.pdf


BRDF usually ignores dependence on wavelength, just "averaging" results into RGB channel
values.


Geant4 Optical
----------------

* flat REFLECTIVITY as function of wavelength, 

  * is there any Fresnel reflectivity increaese as incident angle increases ?






EOU
}
pbs-dir(){ echo $(local-base)/env/graphics/shading/graphics/shading-pbs ; }
pbs-cd(){  cd $(pbs-dir); }
pbs-mate(){ mate $(pbs-dir) ; }
pbs-get(){
   local dir=$(dirname $(pbs-dir)) &&  mkdir -p $dir && cd $dir

}
