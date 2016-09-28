# === func-gen- : presentation/presentation fgp presentation/presentation.bash fgn presentation fgh presentation
presentation-src(){      echo presentation/presentation.bash ; }
presentation-source(){   echo ${BASH_SOURCE:-$(env-home)/$(presentation-src)} ; }
presentation-vi(){       vi $(presentation-source) ; }
presentation-env(){      elocal- ; }
presentation-usage(){ cat << EOU

Presentation preparation
============================

Preparation workflow:

#. change presention name below and create the .txt
#. iterate on presentation by running the below which invokes *presentation-make*
   and *presentation-open* to view local html pages in Safari::

   presentation.sh

#. static .png etc.. are managed within bitbucket static repo, 
   local clone at ~/simoncblyth.bitbucket.org

   * remember not too big, there is 1GB total repo limit 

#. running presentation.sh updates the derived html within
   the static repo clone, will need to "hg add" to begin with


Publishing to remote:

#. update index page as instructed in *bitbucketstatic-vi*
#. push the statics to remote 


Creating retina screencaptures
---------------------------------

* shift-cmd-4 and drag out a marquee, this writes .png file to Desktop

::

   cd ~/env/presentation   ## cd to appropriate directory for the capture

   osx_                    ## precursor define the functions
   osx_ss-cp name          

   ## copy last screencapture from Desktop to corresponding relative dir beneath ~/simoncblyth.bitbucket.org 
   ## this is the local clone of the bitbucket statics repo


Incorporating retina screencaptures
-------------------------------------

::

    simon:presentation blyth$ cd ~/simoncblyth.bitbucket.org/
    simon:simoncblyth.bitbucket.org blyth$ downsize.py env/graphics/ggeoview/PmtInBox-approach.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize env/graphics/ggeoview/PmtInBox-approach.png to create env/graphics/ggeoview/PmtInBox-approach_half.png 2138px_1538px -> 1069px_769px 
    simon:simoncblyth.bitbucket.org blyth$ 



s5 rst underpinning
--------------------

* http://docutils.sourceforge.net/docs/user/slide-shows.html#s5-theme-files

presentations
---------------

g4dae_geometry_exporter.txt
     G4DAE : Export Geant4 Geometry to COLLADA/DAE XML files
     19th Geant4 Collaboration Meeting, Okinawa, Sept 2014

gpu_optical_photon_simulation.txt
     200x Faster Optical Photon Propagation with NuWa + Chroma ?
     Jan 2015

gpu_accelerated_geant4_simulation.txt
     GPU Accelerated Geant4 Simulation with G4DAE and Chroma
     Jan 2015 
     
optical_photon_simulation_with_nvidia_optix.txt
     Optical Photon Simulation with NVIDIA OptiX
     July 2015

optical_photon_simulation_progress.txt
     Opticks : GPU Optical Photon Simulation using NVIDIA OptiX
     Jan 2016


EOU
}
presentation-dir(){ echo $(env-home)/presentation ; }
presentation-cd(){  cd $(presentation-dir); }

presentation-ls(){   presentation-cd ; ls -1t *.txt ; }
presentation-txts(){ presentation-cd ; vi $(presentation-ls) ;  }


#presentation-name(){ echo gpu_accelerated_geant4_simulation ; }
#presentation-name(){ echo optical_photon_simulation_with_nvidia_optix ; }
#presentation-name(){ echo optical_photon_simulation_progress ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_psroc ; }

#presentation-name(){ echo opticks_gpu_optical_photon_simulation_march2016 ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_april2016_gtc ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_may2016_lecospa ; }
#presentation-name(){ echo opticks_gpu_optical_photon_simulation_jul2016_weihai ; }
#presentation-name(){ echo jnu_cmake_ctest ; }
presentation-name(){ echo opticks_gpu_optical_photon_simulation_oct2016_chep ; }

presentation-path(){ echo $(presentation-dir)/$(presentation-name).txt ; }
presentation-export(){
   export PRESENTATION_NAME=$(presentation-name)
}
presentation-edit(){ vi $(presentation-path) ; }
presentation-make(){
   presentation-cd
   presentation-export
   env | grep PRESENTATION
   make $*
}

presentation-remote(){
   echo simoncblyth.bitbucket.org
}

presentation-open(){
   open http://localhost/env/presentation/$(presentation-name).html?page=${1:-0}
} 

presentation-open-remote(){
   open http://$(presentation-remote)/env/presentation/$(presentation-name).html?page=${1:-0}
}

