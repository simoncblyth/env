GTC Preparation
================


Intro
-------

I will introduce Opticks, an optical photon simulation based on OptiX, 
that enables neutrino physics to benefit from 
from massive parallelism  



Dayabay Intro
--------------

*  http://www.symmetrymagazine.org/article/octobernovember-2006/catching-neutrinos-in-china


Described as "liquid onions," each detector will contain three layers. In the
center will be 20 tons of organic liquid scintillator that contains gadolinium,
a heavy metal. Next is a layer of liquid scintillator without gadolinium. The
outer layer uses mineral oil to act as shielding. When an antineutrino
interacts with an atom inside the detector, it produces a positron and a
neutron. The energy from the positron is deposited in the scintillator, which
creates a burst of light. About 30 microseconds later, a second burst of light
is produced as the gadolinium captures the neutron and amplifies the signal.
Photomultiplier tubes that line the mineral-oil-filled outer detector tank
record the light produced in this reaction. Both light flashes must be present
to indicate an electron antineutrino event.



Neutrino Detection
-------------------

* http://t2k-experiment.org/neutrinos/neutrino-detection/

* http://neutrino.physics.berkeley.edu/About.html

* http://neutrino.physics.berkeley.edu/About/DetailedDetector.html

* http://www.scholarpedia.org/article/The_Daya_Bay_Experiment


Potential Users for fast Optical Photon Simulation
---------------------------------------------------

* http://www.opengatecollaboration.org/home

GATE is an advanced opensource software developed by the international OpenGATE
collaboration and dedicated to numerical simulations in medical imaging and
radiotherapy. It currently supports simulations of Emission Tomography
(Positron Emission Tomography - PET and Single Photon Emission Computed
Tomography - SPECT), Computed Tomography (CT), Optical Imaging (Bioluminescence
and Fluorescence) and Radiotherapy experiments. 




Talk coaching 3 
-----------------

Starting : 1 min 
~~~~~~~~~~~~~~~~~~

In a very short time, establish:

* Attention

* Interest

* Credibility 

* Expectations 

  * how my technology works and how this can change the ... industry 
  * show you how Opticks enables particle physics (especially neutrino physics) 
    to start to really benefit from GPUs

Transitions
~~~~~~~~~~~~~

* now that i have introduced ... i will move on to using ...

Concluding : 30s
~~~~~~~~~~~~~~~~~~~~

Signalling the end

* recap : what you said : very high level 

* bookends : the intro and conclusion should be similar

* clincher statements




Abstract
---------


::

    Opticks : Optical Photon Simulation for High Energy Physics with OptiX 

    Opticks is an open source project that integrates high performance GPU 
    ray tracing from NVIDIA OptiX with existing Geant4 toolkit based simulations.
    Advantages of separate optical photon simulation and    
    the approaches developed to integrate it with the general Geant4
    particle simulation are presented. Approaches to minimize overheads
    arising from split are shown.
    Challenges included bringing complex geometries with wavelength
    dependent material and surface properties to the GPU.
    Techniques for visualisation of photon propagations with
    interactive time scrubbing and history selection using OpenGL/OptiX/Thrust
    interoperation and geometry shaders are described.
    Results and demonstrations are shown for the photomultiplier based 
    Daya Bay and JUNO Neutrino detectors. 
    Extrapolation of observed timings with test geometries to multi-GPU workstation
    core counts suggests performance sufficient for optical photon processing time to 
    become effectively zero compared to total simulation time is within reach.



Title Image
-------------

~/simon

jpmt-inside-wide.png 2864x1688

::


    delta:doc blyth$ crop.py --style custom --width 2560 --height 1440 --top 0 --left 0  ~/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide.png
    2016-03-26 12:18:25,123 env.doc.crop INFO     Crop None custom 
    2016-03-26 12:18:25,123 env.doc.crop INFO     cropping /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide.png to create /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop.png 
    2016-03-26 12:18:25,125 env.doc.crop INFO     width 2864 height 1688 cropping to box (0, 0, 2560, 1440) 
    delta:doc blyth$ 





target     1280x720
target*2   2560x1440


PMT pic
-------

* http://neutron.physics.ucsb.edu/docs/High_energy_PMT_TPMO0007E01.pdf

PMT review
------------

* http://www.annualreviews.org/doi/pdf/10.1146/annurev-nucl-102212-170604



Stan Seibert on Chroma
------------------------

* http://on-demand.gputechconf.com/gtc/2013/presentations/S3304-Particle-Physics-With-PyCUDA.pdf
* http://on-demand.gputechconf.com/gtc/2013/video/S3304-Chroma-Photon-Sim-Physics-PyCUDA.mp4



SN1987a
---------

* http://www.astronnewsroom.com/tag/david-malin/

SK
---

* https://en.wikipedia.org/wiki/Kamioka_Observatory
* http://www-sk.icrr.u-tokyo.ac.jp/sk/gallery/wme/PH20-water-withboat-apr23-wm.jpg

The Super-Kamiokande detector is massive, even by particle physics standards.
It consists of 50,000 tons of pure water surrounded by about 11,200
photomultiplier tubes. The detector was again designed as a cylindrical
structure, this time 41.4 m tall and 39.3 m across.

::
 
    curl -L -O http://www-sk.icrr.u-tokyo.ac.jp/sk/gallery/wme/PH20-water-withboat-apr23-wm.jpg

    source 3872x2592
    target 2560x1440

    width  3872 - 2560 = 1312  , 1312/2 = 656
    height 2592 - 1440 = 1152  , 1152/2 = 576


    crop.py --style custom --width 2560 --height 1440 --top 656 --left 576 PH20-water-withboat-apr23-wm.jpg

Central crop decreases image impact::

    delta:tmp blyth$ crop.py --style custom --width 2560 --height 1440 --top 656 --left 576 PH20-water-withboat-apr23-wm.jpg
    2016-03-26 15:02:27,219 env.doc.crop INFO     Crop None custom 
    2016-03-26 15:02:27,220 env.doc.crop INFO     cropping PH20-water-withboat-apr23-wm.jpg to create PH20-water-withboat-apr23-wm_crop.jpg 
    2016-03-26 15:02:27,221 env.doc.crop INFO     width 3872 height 2592 cropping to box (576, 656, 3136, 2096) 
    delta:tmp blyth$ 


Try just trimming the base to give 16:9::

    In [5]: 3872./2592.
    Out[5]: 1.4938271604938271

    In [6]: 3872./16
    Out[6]: 242.0

    In [7]: 3872./16*9
    Out[7]: 2178.0


    delta:tmp blyth$ crop.py --style custom --width 3872 --height 2178 --top 0 --left 0 PH20-water-withboat-apr23-wm.jpg
    2016-03-26 15:12:30,407 env.doc.crop INFO     Crop None custom 
    2016-03-26 15:12:30,407 env.doc.crop INFO     cropping PH20-water-withboat-apr23-wm.jpg to create PH20-water-withboat-apr23-wm_crop.jpg 
    2016-03-26 15:12:30,408 env.doc.crop INFO     width 3872 height 2592 cropping to box (0, 0, 3872, 2178) 

    delta:tmp blyth$ crop.py --style custom --width 3872 --height 2178 --top 0 --left 0 PH20-water-withboat-apr23-wm.jpg --ext .png
    2016-03-26 15:23:26,170 env.doc.crop INFO     Crop None custom 
    2016-03-26 15:23:26,170 env.doc.crop WARNING  converting ext from .jpg to .png 
    2016-03-26 15:23:26,170 env.doc.crop INFO     cropping PH20-water-withboat-apr23-wm.jpg to create PH20-water-withboat-apr23-wm_crop.png 
    2016-03-26 15:23:26,173 env.doc.crop INFO     width 3872 height 2592 cropping to box (0, 0, 3872, 2178) 
    delta:tmp blyth$ 

    3872/2 
    2592/2




G4
---

* https://geant4.web.cern.ch/geant4/results/talks/MC2010/MC2010-Geant4-status.pdf
* https://geant4.web.cern.ch/geant4/results/reports.shtml

* http://geant4.cern.ch/collaboration/workshops.shtml

21st Geant4 Collaboration Meeting, Ferrara (Italy), 12-16 September 2016.


Census of large PMT experiments, existing and planned
-------------------------------------------------------

Make a table of large PMT expts 

* PMT size, type, number
* collaboration dates



MCP-PMT design
* http://ndip.in2p3.fr/ndip11/AGENDA/AGENDA-by-DAY/Presentations/3Wednesday/AM/ID47-Qian.pdf

Simulation of large photomultipliers for experiments in astroparticle physics
* http://arxiv.org/pdf/1001.1283v2.pdf


Amanda, 
IceCube, 
IceTop,
Kamiokande, 
NESTOR, 
NEMO, 
Antares,
MiniBooNE, 
Xenon, 
Baikal GVD
Tunka,
North Auger Observatory

https://www.mpi-hd.mpg.de/hfm/CosmicRay/CosmicRaySites.html

https://en.wikipedia.org/wiki/List_of_neutrino_experiments



DYB
----

* http://dayabay.ihep.ac.cn/pubtalk/pubtalk.html
* http://dayabay.ihep.ac.cn/pubtalk/APS_PMT_WeiliZhong.pdf


Pedro
~~~~~~

* http://dayabay.ihep.ac.cn/cgi-bin/DocDB/ShowDocument?docid=8081
* http://dayabay.ihep.ac.cn/DocDB/0080/008081/004/MasterClass_DayaBay.pdf


Dan
~~~~


* http://dayabay.ihep.ac.cn/DocDB/0085/008506/001/NeutrinosAndDayaBay_DanDwyer_SLAC_29Oct2012.pdf


LiangJan
~~~~~~~~~

* http://dayabay.ihep.ac.cn/DocDB/0052/005277/001/Overview_of_AD_optical_model.pdf



Hammamatsu on SN1987a
----------------------

* http://www.hamamatsu.com/jp/en/technology/projects/exploring_neutrinos/index.html


FT
---

Hammamatsu

* http://markets.ft.com/research/Markets/Tearsheets/Business-profile?s=6965:TYO


Reactor Neutrino expts 
-----------------------

Accelerator Neutrino expts
----------------------------

* https://en.wikipedia.org/wiki/Accelerator_Neutrino_Neutron_Interaction_Experiment


Neutrino Telescopes
--------------------

* KM3Net
* Baikal-GVD
* IceCube/PINGU

* https://en.wikipedia.org/wiki/Deep_Underground_Neutrino_Experiment


Large Water Cerenkov Detectors
--------------------------------

* SNO
* SuperKamiokande/HyperKamiokande


Large Scintillator Detectors
-----------------------------

* 





DYB Images
----------

* http://photos.lbl.gov/albums.php?albumId=141028


* http://photos.lbl.gov/viewphoto.php?source=search&page=&searchField=ALL&searchstring=Daya%20Bay&orient=any&resolution=&resolutionOperand=min&fileSize=&fileSizeOperand=&fileWidth=&fileWidthOperand=min&fileHeight=&fileHeightOperand=min&dateAddedStart=&dateAddedEnd=&dateTakenStart=&dateTakenEnd=&dateExpirStart=&dateExpirEnd=&sort=&sortorder=&linkperpage=20&doccontents=1&albumId=&imageId=5210356&page=39&imagepos=767&sort=&sortorder=



* http://photos.lbl.gov/viewphoto.php?source=search&page=&searchField=ALL&searchstring=Daya%20Bay&orient=&resolution=&resolutionOperand=&fileSize=&fileSizeOperand=&fileWidth=&fileWidthOperand=&fileHeight=&fileHeightOperand=&dateAddedStart=&dateAddedEnd=&dateTakenStart=&dateTakenEnd=&dateExpirStart=&dateExpirEnd=&sort=capturedate&sortorder=&linkperpage=20&doccontents=1&albumId=&imageId=6237550&page=32&imagepos=639&sort=capturedate&sortorder=


Far site WP filling

* http://photos.lbl.gov/viewphoto.php?source=search&page=&searchField=ALL&searchstring=Daya%20Bay&orient=&resolution=&resolutionOperand=&fileSize=&fileSizeOperand=&fileWidth=&fileWidthOperand=&fileHeight=&fileHeightOperand=&dateAddedStart=&dateAddedEnd=&dateTakenStart=&dateTakenEnd=&dateExpirStart=&dateExpirEnd=&sort=capturedate&sortorder=&linkperpage=20&doccontents=1&albumId=&imageId=6237528&page=31&imagepos=620&sort=capturedate&sortorder=


::

    delta:tmp blyth$ crop.py --style custom --ext .png --width 1600 --height 900 --left 0 --top 0 DybFar.jpg
    2016-03-26 17:48:59,443 env.doc.crop INFO     Crop None custom 
    2016-03-26 17:48:59,443 env.doc.crop WARNING  converting ext from .jpg to .png 
    2016-03-26 17:48:59,443 env.doc.crop INFO     cropping DybFar.jpg to create DybFar_crop.png 
    2016-03-26 17:48:59,445 env.doc.crop INFO     width 1600 height 900 cropping to box (0, 0, 1600, 900) 
    delta:tmp blyth$ 
    delta:tmp blyth$ open DybFar_crop.png 




IMPORTANT – READ CAREFULLY: This End User License Agreement (“Agreement") is a
legal agreement between you (in your capacity as an individual and as an agent
for your company, institution or other entity) (collectively, "you" or
"Licensee") and The Regents of the University of California, Department of
Energy contract-operators of the Ernest Orlando Lawrence Berkeley National
Laboratory ("Berkeley Lab").  

Downloading, displaying, using, or copying of the
image  by you or by a third party on your behalf indicates your agreement to be
bound by the terms and conditions of the End User License Agreement and that
you have read and agree to the Copyright Notice, Disclaimers and Usage Terms.

You agree that the image selected by you  may be used for noncommercial,
educational purposes only; no derivative works are allowed, photos of
individuals may only be used for identifying the individual and/or their
research, and attribution and copyright notice is required. 

Please credit the
Lawrence Berkeley National Laboratory and provide the following copyright
notice: "© 2010 The Regents of the University of California, through the
Lawrence Berkeley National Laboratory." 
If you do not agree to these terms and conditions, do not download, 
display or use the image.


http://www.lbl.gov/EndUserLicenseAgreement.html


Dyb Principal
---------------

* http://neutrino.physics.berkeley.edu/About/DetailedDetector.html


Soeren

* http://dayabay.ihep.ac.cn/docs/slides_NuFact2013_sj.pdf


Liquid Scintillator Review
----------------------------

* https://www-opera.desy.de/publications/desyphdseminar/Neutrino_detection_with_liquid_scintillator_Daniel_Bick.pdf


SK Images
----------

::

    Dear Public Affairs Dept,

    I am a Physicist working at the National Taiwan University, Taipei with the
    Daya Bay and JUNO Collaborations.

        http://www-sk.icrr.u-tokyo.ac.jp/sk/gallery/wme/PH13-bottom-yoko-1-wm.JPG

    I would like to use some of your images, such as to above, in an upcoming 
    presentation at the GPU technology conference in San Jose California.

         http://www.gputechconf.com

    My presentation is on an open source  Optical Photon Simulation 
    that I have developed using NVIDIA OptiX ray tracing framework, 
    the abstract of my talk is below.

    A technical presentation of my work is accessible below.

        http://simoncblyth.bitbucket.org/env/presentation/opticks_gpu_optical_photon_simulation_march2016.html

    To make this work accessible to a diverse audience without a background in physics
    I need to provide context and motivation for optical photon simulation.  I think that 
    introducing the audience to the use of photomultiplier tubes in several large physics 
    experiments including Super Kamiokande would be an excellent way 
    to do this, especially due to the beautiful images you have provided at
     
         http://www-sk.icrr.u-tokyo.ac.jp/sk/gallery/index-e.html

    Sincerely,
               Dr Simon C. Blyth


     
    Opticks : Optical Photon Simulation for High Energy Physics with OptiX 

    Opticks is an open source project that integrates high performance GPU 
    ray tracing from NVIDIA OptiX with existing Geant4 toolkit based simulations.
    Advantages of separate optical photon simulation and    
    the approaches developed to integrate it with the general Geant4
    particle simulation are presented. Approaches to minimize overheads
    arising from split are shown.
    Challenges included bringing complex geometries with wavelength
    dependent material and surface properties to the GPU.
    Techniques for visualisation of photon propagations with
    interactive time scrubbing and history selection using OpenGL/OptiX/Thrust
    interoperation and geometry shaders are described.
    Results and demonstrations are shown for the photomultiplier based 
    Daya Bay and JUNO Neutrino detectors. 
    Extrapolation of observed timings with test geometries to multi-GPU workstation
    core counts suggests performance sufficient for optical photon processing time to 
    become effectively zero compared to total simulation time is within reach.


::

    Dear Dr. Simon C. Blyth,

    Thank you so much for your inquiry.
    There is no problem to use our image for your purpose.

    Please credit as
    Kamioka Observatory, ICRR(Institute for Cosmic Ray Research), The University of Tokyo.

    Best regards,
    Yumiko Takenaga



