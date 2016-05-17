LeCosPA May 2016
==================


Practicalities
---------------

::

    Seminar : Leung Center for Cosmology and Particle Astrophysics (LeCosPA) 
    of the National Taiwan University (NTU) 

    Our seminar will be a 50 minutes long talk which starts at 11:00 AM every Monday. 
    For the Spring 2016 semester, the seminar series starts from Mar 7 to
    July 25. You can find the available dates in the following link:

    http://wiki.lecospa.ntu.edu.tw/index.php/2016_Spring_Seminar_List

    Location : Rm 812, Astro-Math Building, NTU
    Time : 11:00-12:00, Monday


IMB
----

* https://deepblue.lib.umich.edu/bitstream/handle/2027.42/31079/0000756.pdf?sequence=1   


Neutrino History
-----------------

* http://t2k-experiment.org/neutrinos/neutrino-detection/


Flow
----------

::

    00 : Opticks : Optical Photon Simulation for Particle Physics with NVIDIA OptiX
        
        Opticks is an an open source project I developed that applies 
        GPU acceleration to optical photon simulation.
        
    01 : opticks benefits 

        The benefits are dramatic... the more photons the more benefit 

        Large photomultiplier based neutrino detectors, such as the JUNO detector 
        visualized here, can benefit greatly from ultrafast optical photon simulation.
        This image visualizes scintillation and cerenkov photons coming 
        from a cosmic ray muon crossing the JUNO detector.

    02 : outline 

        I developed Opticks for Daya Bay and JUNO : which are PMT based neutrino detectors
        I'll describe some crucial aspects of PMT based detectors 
        and highlight a few physics results and future prospects. 

        Then I'll describe the standard approach to simulation, how GPU acceleration
        can help, the practical aspects of doing so, then give validation and 
        performance results before ending with a short demonstration video.

        03 : photomultiplier tube based detectors : primary tool of neutrino physics 

             * history of neutrinos : dominated by results from PMT based experiments
             * future of neutrinos : looks set to continue that 

             I'll talk more about the ones in red.

             All of these required an understanding of optical photon generation and propagaton 

             Drastic simulation speedup is not just a convenience, short development
             cycles make you more efficient, and that enables progress in the form 
             of improved understanding.

        03-04 : PMTs

             PMTs can detect single photons with a large angular coverage,  
             thanks to the photoelectric effect with very thin Bialkali photocathodes 
             evaporated onto the inside of the glass bulb, the single photoelectron is 
             amplified through ...

        05-06 : SN1987A 

              This is an artists impression ...

              A burst of 24 neutrinos was detected within 13 seconds are three 
              neutrino detectors in Japan, US and Russia.
              This was the first and so far only extra-solar neutrino astronomy 
              observation ... which was also extra-galactic coming from the nearby 
              Large Megallanic Cloud galaxy.

              * https://indico.lal.in2p3.fr/event/a05162/session/s21/contribution/s21t2/material/0/0.pdf


        07-10 : Super-Kamiokande : Cerenkov Light, rings e-like mu-like, upward downward
 
              Next example of a PMT detector is Super-Kamiokande, this is a water cherenkov
              detector, during operation the cavern is filled with water that is instrumented
              with 20 inch PMTs



        11 : dayabay reactor neutrino expt, far site 

        Each of those 4 modules is an antineutrino detector or AD.  

        12 : daya bay far site 2  : AD design 

        Three liquids separated by inner and outer acrylic vessels

        * Gadolinium doped liquid scintillator target
        * surrounded by undoped LS gamma catcher 
        * mineral oil 

        * PMTs mounted in the mineral oil
        * calibration system sits on the lid, radioactive sources and LEDs
          are deployed into the liquids and data is taken at various positions 

        13 : daya bay pmt wall photo 1 

        View inside the AD, showing the PMTs front faces and acrylic vessel 

        14 : daya bay antineutrino detection via inverse beta decay 1 

        So how to detect neutrinos. 

        * Electron anti-neutrino interacts with proton of target via inverse beta decay 
          producing positron and neutron.

          * positron promptly annilates producing gammas, from the kinematics
            the prompt energy provides an estimate of antineutrino energy  

          * neutron captured on Gadolinium (which has very high neutron capture cross-section) 
            or proton with a characteristic times, resulting in delayed 8 MeV across 
            multiple gammas 

          Coincidence of prompt and delayed energy is a very distinct signal, making 
          for excellent background reduction.

    15 : daya bay antineutrino detection via inverse beta decay 2 

          Sounds straightforward, but there are difficulties in the details 
          of the energy response.

    16 : daya bay energy response model : fit to calibration data 1 

          Energy response model uses 4 stages,  


04 : photomultiplier tubes (pmts) 
05 : photomultiplier tube operation 


06 : kamiokande ii 1 
07 : kamiokande ii 3 
08 : super-kamiokande pmts 1 
09 : super-kamiokande pmts 4 
10 : super-kamiokande pmts 3 
11 : super-kamiokande pmts 2 
12 : dayabay reactor neutrino expt, far site 
13 : daya bay far site 2 
14 : daya bay far site 3 
15 : daya bay pmt wall photo 1 
16 : daya bay antineutrino detection via inverse beta decay 1 
17 : daya bay antineutrino detection via inverse beta decay 2 
18 : daya bay ngd analysis : most precise theta13 
19 : daya bay : e(prompt) -> detector unfolded reactor antineutrino spectrum 
20 : jiangmen underground neutrino observatory (juno) 
21 : juno : neutrino mass heirarchy (mh) determination 
22 : juno : a multi-purpose neutrino observatory 
23 : neutrino physics with juno 
24 : liquid scintillator (ls) particle dependent non-linear light yield 
25 : liquid scintillator ls ; ternary (lab/ppo/bis-msb) optimization 
26 : geant4 : monte carlo simulation toolkit 
27 : geant4 : monte carlo simulation toolkit generality 
28 : but, there is a problem... 
29 : jpmt before contact 2 
30 : ray traced realistic image synthesis and optical photon simulation 
31 : nvidia optix 1 
32 : nvidia optix 2 
33 : brief history of gpu optical photon simulation development 
34 : migrating geometry from geant4 to opticks 
35 : large geometry/event techniques 
36 : geometry modelling : tesselated vs analytic photomultiplier tubes 
37 : tesselated photomultiplier : unrealistic disco ball effect 
38 : analytic pmts together with triangulated geometry 
39 : visualizing an optical photon simulation 
40 : compare opticks/geant4 simulations with simple lights/geometries 
41 : opticks/geant4 rainbow step sequence comparison 
42 : rainbow spectrum for 1st six bows 
43 : 1m rainbow s-polarized, comparison opticks/geant4 
44 : pmtinbox after 1 
45 : pmtinbox after 2 
46 : pmt opticks/geant4 step distribution comparison to bt [sd] 
47 : pmt opticks/geant4 step distribution comparison : chi2/ndf 
48 : photon propagation times geant4 cf opticks 
49 : summary 
50 : youtube video 
51 : youku video 
52 : jpmt wide 
53 : jpmt after contact 
54 : jpmt approach 
55 : jpmt arrival 
56 : jpmt inside outside 
57 : optix experience 





What to add to GTC talk ?
---------------------------


Addition ideas

* intro OptiX
* what is ray tracing 

Studies were Opticks will be particularly helpful


Strategy
---------

* need to plug and provide refs to DYB recent papers
* similary for JUNO Yellow Book  

* need to show how optical photon simulation is relevant to DYB and JUNO


Survey optical photons and where they fit in 
----------------------------------------------
    
* optically coupled scintillator and PMTs
* energy resolution and time resolution modelling of neutrino detectors
* scintillator modelling, Birks Quenching, how many photons 
* scintillator reemission spectrum
* Gd doping effect

* scint composition: Gd-TMHA + LAB + 3g/L PPO + 15mg/L bis-MSB
* http://dayabay.ihep.ac.cn/pubtalk/Zhang-INFO11.pdf


* Monte Carlo

  * detector design optimization
  * multiple fluors, JUNO scintillator modelling
  * PMT characteristics

  * energy resolution, esp for JUNO PMT resolution requirements, usefulness of sPMT 

    * energy resolution numbers for Daya Bay, JUNO
    * transit time 



SuperK
-------

* https://www.nobelprize.org/nobel_prizes/physics/laureates/2015/advanced-physicsprize2015.pdf


Scintillation in practice
----------------------------

* :google:`theory and practice of scintillation counting`
* http://www-physics.lbl.gov/~spieler/physics_198_notes/PDF/III-Scint.pdf

* http://web.utk.edu/~kamyshko/P627/L25.pdf
* http://www.lsc-international.org/conf/pfiles/horrocks_1974_complete.pdf

  nice intro to complexity of energy transfer in scintillators

* http://dayabay.ihep.ac.cn/docs/0701029.pdf

  DYB Proposal : includes discussion of development of the the liquids

 * http://dayabay.ihep.ac.cn/cgi-bin/DocDB/ShowDocument?docid=10780

Birks Law in Geant4
--------------------

* http://irtg.physi.uni-heidelberg.de/activities/schools/fall_school_2010/A.Tadday.Application.of.Birks.low.of.nonlinearity.in.GEANT.pdf

Application of Birks' law of scintillator nonlinearity in Geant4
Alexander Tadday Kirchhoff Institute for Physics Heidelberg University


DYB Overview
------------

* http://dayabay.ihep.ac.cn/DocDB/0107/010780/002/DayaBay_NuclPhysB.pdf
  
  Cao and Luk



DYB Energy Model
------------------

Soren
~~~~~~~


* http://dayabay.ihep.ac.cn/DocDB/0100/010044/006/slides_APC.pdf

  Talk : nice figure describing energy model

* http://dayabay.ihep.ac.cn/DocDB/0098/009826/002/summary_energy_model_update.pdf

  Soren on: energy model   

* http://dayabay.ihep.ac.cn/DocDB/0103/010330/002/note_revised_energy_model_v2.pdf

  Soren on: proposed revised energy model that incorporates corrections of
  recently identified biases in gamma calibration data due to energy losses in
  the source shieldings and optical shadowing by the source enclosures and weights.

* http://dayabay.ihep.ac.cn/DocDB/0096/009641/005/poster_neutrino14_energy_response_v3.pdf


DYB Optical
~~~~~~~~~~~~~~~~~~~~~~~

Liangjan

* http://dayabay.ihep.ac.cn/DocDB/0052/005277/001/Overview_of_AD_optical_model.pdf
* http://dayabay.ihep.ac.cn/DocDB/0044/004439/001/OpticalModel_simu_ana_workshop_Dec19_LJWen.pdf
* http://dayabay.ihep.ac.cn/cgi-bin/DocDB/ListBy?authorid=31  

Doc 4525, by H,L Xiao

* http://dayabay.ihep.ac.cn/cgi-bin/DocDB/ListBy?authorid=256  
* http://dayabay.ihep.ac.cn/cgi-bin/DocDB/ShowDocument?docid=4525

Study of absorption and re-emission processes in a ternary liquid scintillation system

* http://dayabay.ihep.ac.cn/DocDB/0035/003537/002/fqe_measurement.pdf

Emission spectrum and fluorescence quantum efficiency measurements
Hua-Lin Xiao
Xiao-Bo Li
April 30, 2009


DYB Talks
---------

* http://dayabay.ihep.ac.cn/DocDB/0107/010745/002/grassi_dyb.pdf Marco from Feb 2016
* http://dayabay.ihep.ac.cn/DocDB/0104/010448/006/Dayabay_wangzhe_Taup15.pdf Zhe from Sep 2015


DYB Papers
-----------

* http://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.061801

  Measurement of the Reactor Antineutrino Flux and Spectrum at Daya Bay


Moriond EW2016 ZhangYiming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dayabay.ihep.ac.cn/DocDB/0108/010818/003/Moriond_EW2016_ZhangYiming_v3.pdf

Non-linearity semi-empirical model, 

* eqn for fq fc



DYB JUNO Combined Talks
------------------------

Wei Wang

* http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=1069



JUNO 
-----

* http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/DocumentDatabase
* http://juno.ihep.ac.cn/mediawiki/index.php/InternalWeb


JUNO : A multi-purpose neutrino observatory

* http://www.taup-conference.to.infn.it/2015/day2/parallel/nua/2_wurm.pdf

* http://dayabay.ihep.ac.cn/DocDB/0099/009988/001/Poster.pdf

 
  DYB and JUNO scintillator poster



JUNO Talk
~~~~~~~~~~~

* http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=803

Talk at Neutel2015, Jun Cao


SPMT
~~~~~~

* http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ListBy?authorid=53
* http://juno.ihep.ac.cn/Dev_DocDB/0014/001476/003/spmt%20v2.pdf
* http://juno.ihep.ac.cn/Dev_DocDB/0008/000859/001/JUNOMultiCalo_Anatael_150420.pdf



Neutrino Physics with JUNO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://arxiv.org/pdf/1507.05613v2.pdf Neutrino Physics with JUNO
* ~/Desktop/juno_1507.05613v2.pdf

p192 - 193 : Monte Carlo Simulation

Most of the properties of the detector materials are borrowed from the Daya Bay
experiment. The elemental concentrations of the liquid scintillator were
measured and incorporated into the MC. All relevant optical properties of the
detector components are derived from measurements, including refractive indices
of the liquids as well as the acrylic or nylon components, time constants and
photon emission spectra of the liquid scintillator, and the reflectivity of the
detector materials. Photon absorption and re-emission processes in liquid
scintillator are modeled based on measure- ments in order to properly simulate
the propagation of scintillation photons and contributions from Cherenkov
process.


To achieve an energy resolution of 3%/√E, 17746 20-inch PMTs corresponding to 77% 
photocathode coverage are used. The quantum efficiency is set to be 35%. 
The attenuation length of the LS is set to be 20 m, 
larger than 15 m for the Daya Bay. Recently we have measured
the rayleigh scattering length of the liquid scintillator to be ∼30 m. The
light yield of the liquid scintillator is tuned to the data of the Daya Bay
detector. We find that the 3%/√E energy resolution is achieved with the above
settings. Therefore, they served as the requirements for the JUNO detector
design. The baseline parameters of the JUNO MC are shown in Tab. 13-3, as well
as in the following.  Detector dimensions. The scintillator volume is 35.4 m in
diameter, surrounded by a buffer medium with a thickness of 1.5 m, either water
in the acrylic ball option or non-scintillation oil in the balloon option. PMTs
are assumed to be 20-inch PMTs, with their bulk center located at 19.5 m in
radius. The number of PMTs is 17746. The water pool, as the cherenkov detector
for muons, is a cylinder of 42.5 m in diameter and 42.5 m in height.  

**Light emission.** 

The light yield of the liquid scintillator is about 10000 photons per
MeV. The exact value may vary by the order of 10% depending on the scintillator
solvent and fluor concentrations. For different settings in the simulation, we
normalize the light yield to the response of the Daya Bay detector. 
The light output also depends on the energy loss rate dE/dx of the ionizing particle, 
resulting in a quenched visible energy. This effect is taken into account by the
Birks’ law,

The light is emitted following a time profile described by a superposition of
exponential decays. The simulation uses a description of three components with
time constants τ1, τ2, and τ3, respectively.


**Light propagation.**

The scintillation light is produced in a relatively broad span of wavelengths. 
The emission spectrum of liquid scintillator is from the Daya Bay measurements. The
photon absorption, re-emission, and Rayleigh scattering are simulated.


**Light detection.** 

The baseline design for JUNO assumes an photocathode coverage of 77% and 35% 
peak quantum efficiency of the PMTs. The quantum efficiency spectrum is scaled from the Daya Bay PMTs.

Based on these configurations, the p.e. yield per deposited energy can be
obtained as a function of the vertex position inside the target volume. The
p.e. yield significantly depends on vertex position, the transparency of
liquid scintillator container, and the refractive indices of liquid materials,
as shown in Figure 13-3.




JUNO Energy Resolution Requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://juno.ihep.ac.cn/Dev_DocDB/0010/001077/004/JUNO%4017lom.pdf

To achieve the primary goal of the MH determination, an unprecedented energy resolution of 3%/Evis 
is a critical parameter in the experimental design, which requires: 

* PMT photocathode coverage ≥ 75%, 
* PMT quantum efficiency ≥ 30%@400 nm 
* PMT collection efficiency ≥ 90% [19, 20]
* LS attenuation length ≥ 20 m at 430 nm, 
  which is equivalent to an absorption length of 60 m 
  with a Rayleigh scattering length of 30 m


JUNO Scinillator Model
~~~~~~~~~~~~~~~~~~~~~~~~

* http://juno.ihep.ac.cn/Dev_DocDB/0010/001049/001/lt-JUNO-201507-detsim-20150716-b23.pdf

  Xinying Li, Doc 872, 880, 997


Xinying Li 
~~~~~~~~~~~

* http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ListBy?authorid=62

* http://juno.ihep.ac.cn/Dev_DocDB/0012/001293/002/main-20160128-a034a83.pdf

* http://juno.ihep.ac.cn/Dev_DocDB/0013/001342/001/New_Optical_model.pdf

  Li xinying (1) ,Wen liangjian(2), Deng ziyan(2) 
  JUNO  

  Current Optical Model:

  LS: (solvent:LAB; PPO 3g/L; bis-MSB 15mg/L) 

  * one emission spectrum (bisMSB)
  * one absorption spectrum
  * one assumed re-emission probability curve. Re-emission spectrum is still bisMSB.

  Why consider a new optical model

  * Can make a better understanding of the optical process.
  * The influence on the central detector at different concentrations of PPO and bis-MSB. 
    Optimization of liquid scintillation formula.
  * The influence on hit time at different liquid scintillation formula.

  New Optical Model

  Emission : Use PPO emission spectrum as the origin excited emission spectrum 
  when charge particle interacts with LS. 

  Absorption
  For LS, both PPO and bis-MSB exist. 
  Supposing we have measured the molar attenuation coefficient for PPO and bis-MSB(From Ding Xuefeng), 
  then we can calculate the probability for absorption by PPO or by bis-MSB at different concentration.



Liquid Scintillator
~~~~~~~~~~~~~~~~~~~~

* :google:`fluor PPO and wavelength shifter bis-MSB`




