
Meeting the Challenge of JUNO Simulation with Opticks GPU Optical Photon Simulation
=======================================================================================

I am delighted to have been awarded the CAS PIFI fellowship at IHEP 
as it has enabled me to apply Opticks to the JUNO neutrino detector simulation.
As JUNO will benefit the most from Opticks it is particularly appropriate 
for Opticks development to be done at IHEP, where JUNO detector experts 
are readily accessible and an NVIDIA GPU cluster on which to 
perform simulation production runs is in preparation.  

The volume of the JUNO liquid scintillator will be the largest in the world producing
many millions of optical photons from cosmic ray muon events, causing
extreme computational and memory costs for simulation. 
Opticks is an open source package that I developed which enables Geant4 based 
optical photon simulations to benefit from the industry leading performance of 
the NVIDIA OptiX GPU ray tracing engine, with speedup factors expected to exceed 1000x 
traditional serial approaches with recent GPUs.

The most rewarding work I have done during my PIFI fellowship is the development
of an entirely new direct geometry translation approach for Opticks that converts 
the standard Geant4 detector geometry directly into the Opticks equivalent 
and uploads it to the GPU fully automatically, avoiding multiple manual stages of processing 
for each geometry. This direct approach enables convenient embedded use of Opticks, bringing 
Opticks acceleration to any detector geometry with a minimum of code changes. 
Greatly enhanced ease-of-use is particularly rewarding as it will lead to 
more experiments adopting Opticks. By providing less resource intensive simulation
results to more people Opticks becomes more valuable.

Optical photon simulation on the GPU requires that the full geometry information
is translated into an appropriate form and serialized for upload to the GPU. 
A natural side benefit is high performance GPU accelerated 3D visualizations 
of geometry and event data, that have enabled me to create animations depicting 
flying through the JUNO detector geometry and observing photon propagations.
These animations convey the principals of JUNO detector operation in an engaging 
manner that is impressive to wide audiences.

The wide applicability of Opticks beyond JUNO has resulted in considerable interest 
from neutrino and dark matter search experiments and from the Geant4 Collaboration. 
The interest has also resulted in an invitation to present a keynote talk entitled 
"Meeting the Challenge of JUNO Simulation with Opticks GPU Optical Photon Acceleration”
within the plenary session of the CHEP 2019 conference in November.

The scale of the JUNO experiment with many thousands of PMTs brings new challenges
across all areas of simulation, reconstruction, calibration and analysis.
There are many opportunities for GPU parallel processing to provide 
large performance benefits across all these areas, as clearly demonstrated for 
simulation and as shown already by preliminary reconstruction studies. 
I look forward to future collaboration with CAS colleagues at IHEP 
continuing to apply my experience of GPU workflows across all these areas.

