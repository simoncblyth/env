cepc-env(){ echo -n ; }
cepc-vi(){ vi $BASH_SOURCE ; }

cepc-notes(){ cat << EON
CEPC 
=====

The 2023 international workshop on the high energy Circular Electron-Positron
Collider (CEPC) will take place at Nanjing, Oct 23-27, 2023. The opening
session will be held in the Science and Technology Building of Nanjing
University at its Gulou Campus. All other sessions will be at the Grand Hotel
Nanjing.

The abstract submission deadline is Sept 16, 2023. The registration deadline is
Oct 1, 2023. 


Presentation : 30 mins : What to include ?
---------------------------------------------

Its in "Offline and Software" session : so focus on software aspects. 


CEPC Design 
-------------

* https://arxiv.org/ftp/arxiv/papers/1811/1811.10545.pdf
* ~/opticks_refs/cepc_design_report_phys_detector_1811.10545.pdf

p192
particle flow algorithm (PFA [1]) is a very promising approach to achieve the
un- precedented jet energy resolution of 3–4%. 

The basic idea of the PFA is to make use of the optimal detector subsystem to
determine the energy/momentum of each particle in a jet. An essential
prerequisite for realization of this idea is to distinguish among energy de-
posits of individual particles from a jet in the calorimetry system. High,
three-dimensional spatial granularity is required for the calorimetry system to
achieve this. Therefore, PFA calorimeters feature finely segmented,
three-dimensional granularity and compact, spa- tially separated, particle
showers to facilitate the reconstruction and identification of ev- ery single
particle shower in a jet. It is for this feature PFA calorimeters are usually
also called imaging calorimeters. A PFA calorimetry system generally consists
of an electro- magnetic calorimeter (ECAL), optimized for measurements of
photons and electrons, and a hadronic calorimeter (HCAL) to measure hadronic
showers.



p243

The dual-readout method avoids these limitations by directly mea- suring fem on
an event-by-event basis. The showers are sampled through two independent
processes,namelyscintillation(S)andCˇerenkov(C)lightemissions.Theformerissen-
sitive to all ionizing particles, while the latter is produced by highly
relativistic particles only, almost exclusively found inside the em shower
component. By combining the two measurements, energy and fem of each shower can
be simultaneously reconstructed. The performance in hadronic calorimetry may be
boosted toward its ultimate limit.

p254

In the simulations, the process of generation and propagation of the
scintillation light was switched off and the energy deposited in the fibers was
taken as signal since this does not introduce any bias to the detector
performance. This statement does not apply
totheCˇerenkovphotonsforwhichaparameterizationthatconvolutestheeffectoflight
attenuation, angular acceptance and PDE, was introduced.




Monolithic Active Pixel Sensors (MAPS) for high precision tracker and high granularity calorimetry



The simulation of the Dual-Readout Calorimeter for future collider experiments

* Sanghyun Ko (sanghyun.ko@cern.ch) Seoul National University

* https://indico.bnl.gov/event/11918/contributions/50512/attachments/35032/56968/210608_shKo_EIC.pdf
* ~/opticks_refs/simulation_of_dual_readout_calorimeter_210608_shKo_EIC.pdf

Several slides::

   Speeding up optical photon tracking


Umbrella CEPC LoI
~~~~~~~~~~~~~~~~~~~

* https://www.snowmass21.org/docs/files/summaries/EF/SNOWMASS21-EF1_EF4-IF9_IF0-260.pdf
* ~/opticks_refs/CEPC_Detectors_LoI_LIST_SNOWMASS21-EF1_EF4-IF9_IF0-260.pdf 

IDEA detector LoI : Innovative Detector for an Electron-positron Accelerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.snowmass21.org/docs/files/summaries/EF/SNOWMASS21-EF1_EF4-IF3_IF6-096.pdf
* ~/opticks_refs/IDEA_LoI_SNOWMASS21-EF1_EF4-IF3_IF6-096.pdf

dual-readout (DR) calorimeter. The DR calorimeter has the particularity
of simultaneously measuring the electromagnetic and hadronic components of the showers origi-
nated in the calorimeter volume. The DR calorimeter uses 1 mm diameter alternate scintillating and
Cerenkov fibers, read at the back end by SiPMs. The DR calorimeter provides an excellent energy
resolution, of the order of 30%/ √E, on the measurement of hadronic jets. The electromagnetic
energy resolution is of the order of 10%/ √E, while maintaining a high granularity needed to disen-
tangle close-by shower pairs from neutral pion decays.



* https://www.osti.gov/servlets/purl/1541412
* ~/opticks_refs/dual_readout_calorimetr_1712.05494.pdf

dual readout calorimetryt

he passive medium is usually a high-density material, 
such as iron, copper, lead or uranium. 
The active medium generates the light or charge 
that forms the basis for the signals from such a calorimeter.

...
An alternative approach to eliminate the effects of
the fluctuations in the em shower fraction, which domi-
nate the hadronic energy resolution of non-compensating

calorimeters, is to measure fem f_or each event. It turns
out that the ˇCerenkov mechanism provides unique op-
portunities to achieve this.
Calorimeters that use ˇCerenkov light as signal source
are, for all practical purposes, only responding to the
em fraction of hadronic showers [15]. This is because
the electrons/positrons through which the energy is de-
posited in the em shower component are relativistic down
to energies of only ∼200 keV. On the other hand, most
of the non-em energy in hadron showers is deposited by
non-relativistic protons generated in nuclear reactions.
Such protons do generate signals in active media such as
plastic scintillators or liquid argon. By comparing the
relative strengths of the signals representing the visible
deposited energy and the ˇCerenkov light produced in the
shower absorption process, the em shower fraction can
be determined and the total shower energy can be recon-
structed using the known e/h value(s) of the calorimeter.
This is the essence of what has become known as dual-
readout calorimetry.


...

A dual-readout calorimeter produces two types of sig-
nals for the showers developing in it, a scintillation sig-
nal (S) and a ˇCerenkov signal (C). Both signals can
be calibrated with electrons of known energy E, so that
〈S〉 = 〈C〉 = E for em showers, and the calorimeter re-
sponse to em showers, Rem = 〈S〉/E = 〈C〉/E = 1. 

The dual-readout method works thanks to the fact that
(e/h)S != (e/h)C . The larger the difference between both
values, the better. The em shower fraction fem and the
shower energy E can be found by solving Equations 6, us-
ing the measured values of the scintillation and ˇCerenkov
signals and the known e/h ratios of the ˇCerenkov and
scintillator calorimeter structures. We will 
...

The instrument
they built became known as the DREAM calorimeter.
As before, the two active media were scintillating fibers
which measured the visible energy, while clear, undoped
fibers measured the generated ˇCerenkov light. Copper
was chosen as the absorber material. The basic element
of this detector (see Figure 14) was an extruded copper
rod, 2 meters long and 4×4 mm2 in cross section. This
rod was hollow, the central cylinder had a diameter of
2.5 mm. In this hole were inserted seven optical fibers

* https://www.mdpi.com/2410-390X/6/3/36
* 25 Years of Dual-Readout Calorimetry by Richard Wigmans

On the other hand, if the em fraction is low, the energy that leaks out of the
thin calorimeter is relatively large. It was already known from the prototype
studies for the CMS very-forward calorimeter, which uses quartz fibers as
active material, that the Čerenkov light produced in these fibers is
overwhelmingly generated by the em components of the hadron showers. For these
reasons, simultaneous detection of Čerenkov light and scintillation light (a
measure for 𝑑𝐸/𝑑𝑥) would provide not only information about the energy
deposited in the calorimeter, but also on the relative fraction of em shower
energy and, therefore, about the undetected energy leaking out.

We tested the validity of these ideas (Paper 2, Table 1) with the calorimeter
depicted in Figure 3. The absorber material was lead, 39 plates, each 6.4 mm
thick, for a total depth of 1.4 𝜆int. In between these plates, layers of
ribbons of fibers were inserted. These fibers were alternatingly made of
plastic scintillator and quartz. The fibers from each layer were read out by
small photomultiplier tubes. As shown in the figure, these PMTs were arranged
in such a way that x-y granularity was achieved for both types of readout.
**Essentially, in this way we constructed two calorimeters that provided
completely independent scintillator (S) and Čerenkov (Q) signals from the same
events.**

* https://www.mdpi.com/2410-390X/6/4/59
  SiPMs for Dual-Readout Calorimetry
  by Romualdo Santoro
  on behalf of the IDEA Dual-Readout Group 

Hadronic showers develop both an electromagnetic and a hadronic components
which are usually detected with a different response (non-compensation). As a
result, the fluctuations among the two components constitute one of the most
limiting factors for the hadronic energy resolution. Dual readout is a
calorimetric technique able to overcome the limits due to non-compensation by
simultaneously detecting scintillation and Cherenkov lights. Scintillating
photons provide a signal related to the energy deposition in the calorimeter by
all ionising particles, while Cherenkov photons provide a signal almost
exclusively related to the shower electromagnetic component. In fact, by
looking at the two independent signals, it is possible to measure, event by
event, the electromagnetic shower component and to properly reconstruct the
primary hadron energy.



* https://www.snowmass21.org/docs/files/summaries/IF/SNOWMASS21-IF6_IF0-176.pdf
* Particle Flow Calorimeters for the Circular Electron Positron Collider

A promising approach capable to achieve a jet energy resolution of 3-4% is Particle Flow Algo-
rithm (PFA) [4, 5]. **The basic idea of the PFA is to make full use of the inner tracker, electromagnetic
and hadronic calorimeters to determine the energy/momentum of each particle in a jet. A finely
segmented and compact calorimetry system with high granularity in three dimensions is crucial to
the reconstruction and identification of every single particle shower in a jet, and hence to realize
the PFA**. The CEPC baseline detector concept includes a PFA calorimetry system consisting of
an electro-magnetic calorimeter and a hadronic one both with extremely high granularity. Worldwide
studies have been carried out within the CALICE collaboration [6] to develop compact PFA calorime-
ters. Several prototypes with high granularity using different technologies have been developed, and
exposed to particle beams to gain in-depth understanding of the PFA calorimetry performance [4,5].
There are still various detector technology options that need to be further explored to address chal-
lenges from stringent performance requirements for the CEPC detector de

* https://www.snowmass21.org/docs/files/summaries/IF/SNOWMASS21-IF6_IF0_Yong_Liu-064.pdf
* High-Granularity Crystal Calorimetry Letter of Intent




CEPC optical photons
----------------------

* https://indico.cern.ch/event/915715/contributions/3850280/attachments/2036905/3410659/200512_shKo.pdf

  Fast optical photon transportation in GEANT4
  Sanghyun Ko
  12 May 2020 

* https://indico.ihep.ac.cn/event/16585/contributions/49055/attachments/23377/26500/Research_progress_of_glass_scintillator_for_CEPC-V3.0.pdf

  Research Progress of The Glass Scintillator for CEPC
  Qian Sen

* https://agenda.infn.it/event/28874/contributions/169546/attachments/93698/129318/20220708_A%20novel%20high-granularity%20crystal%20calorimeter.pdf

  A novel high-granularity crystal calorimeter
  Baohua Qi





scintillating and cerenkov fibers
-----------------------------------

Dual Cherenkov and Scintillation Response to High-Energy Electrons of Rare-Earth-Doped Silica Fibers
Francesca Cova, Marco T. Lucchini, Kristof Pauwels, Etiennette Auffray, Norberto Chiodini, Mauro Fasoli, and Anna Vedda
Phys. Rev. Applied 11, 024036 – Published 14 February 2019 



Registration
--------------

* https://indico.ihep.ac.cn/event/19316/registrations/1581/

Arrival: Oct 22, 2023 (Sun)
Departure: Oct 28, 2023 (Sat)
Accommodation: Nanjing Grand Hotel (Single Room) ￥450/night 

Registration fee:
- 2000 CNY/person for staff and postdocs
- 1000 CNY/person for students
- No registration fee for online participants



CEPC Talk : Fri Oct 29 : 09:00-09:30 
---------------------------------------

* https://indico.ihep.ac.cn/event/19316/timetable/
* https://indico.ihep.ac.cn/event/19316/sessions/12209/#20231027


Opticks : GPU Optical Photon Simulation via NVIDIA OptiX
CEPC Room 4, GrandHotelNanjing
09:00 - 09:30


2023 CEPC Workshop in Nanjing
-----------------------------

The 2023 international workshop on the high energy Circular Electron-Positron
Collider (CEPC) will take place at Nanjing, Oct 23-27, 2023. Detailed
information can be found at https://indico.ihep.ac.cn/event/19316 The deadline
of registration is Oct 1, 2023.

Please follow and register for the 2023 CEPC Workshop in Nanjing. Please note
the deadline for submission of abstracts. Looking forward to your
participation.


https://indico.ihep.ac.cn/event/19316/

The abstract submission deadline is Sept 16, 2023.

https://indico.ihep.ac.cn/event/19316/abstracts/

Your abstract 'Opticks : GPU Optical Photon Simulation via NVIDIA OptiX' has
been successfully submitted. It is registered with the number #62. You will be
notified by email with the submission details. 



EON
}
cepc-cd(){ cd $(env-home)/presentation/cepc ; }
cepc-wc(){ 
   cepc-cd
   wc -w *abstract.tex
}



