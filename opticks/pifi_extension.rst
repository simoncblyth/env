PIFI Extension : Opticks Beyond Simulation
=============================================

The scale of the JUNO experiment together with the level of understanding 
necessary to achieve the unprecedented energy resolution required to determine
the neutrino mass hierarchy brings new challenges to all areas of JUNO data processing.
GPU performance continues to grow rapidly, however to harness this power it is
necessary to make a fundamental change to a massively parallel approach to data processing. 
My experience with Opticks has demonstrated huge performance gains for simulation.
I believe there are many opportunities for parallel processing, such as from 
the many thousands of PMTs, that are ripe for achieving large performance benefits 
across all areas of reconstruction, calibration and analysis.  
The activities I below propose for an extension of my PIFI work seek to 
realise this vision by bringing my GPU development experience to bear upon
all areas of JUNO software development. 


Ensure that Opticks simulation gets into production within JUNO and in other experiments
----------------------------------------------------------------------------------------------

Precisely when Opticks simulation gets into production within JUNO 
depends on the problems encountered which cannot be predicted.
Also technicalities of batch running of GPU jobs have yet to 
be determined. Nevertheless extending my PIFI work will 
ensure that the benefits of Opticks GPU acceleration 
will become available to JUNO.

Opticks was designed to solve the optical photon simulation problem 
of neutrino detectors such as JUNO, however any simulation that is 
limited by optical photons can benefit greatly. 
Physicists from several neutrino and dark matter search experiments 
and from industry have discovered Opticks mostly via its web 
presence and have started to evaluate and use Opticks.

I wish to encourage further adoption of Opticks by publicising it through 
conference presentations and publications as well as blog posts and videos
and the establishment of a user forum.
Incorporation of an Opticks extended optical example within the Geant4 distribution 
will be an important milestone as it will ensure that everyone interested 
in optical photon simulation with Geant4 will be made aware of Opticks. 
Following discussions with Geant4 members the general structure of an example is 
agreed upon and I have a prototype example in development. I think it will be 
appropriate to incorporate this within the Geant4 distribution after Opticks
reaches production level within JUNO. 
  
Application of GPU massive parallelism across all areas of JUNO data processing
-----------------------------------------------------------------------------------

The experience I have gained of CUDA and Thrust from working with millions 
of optical photons has made me very aware of the huge benefits of 
GPU massive parallelism. Very few people within HEP have experience of GPU techniques 
resulting in very little application of this new technology within the field, despite
the ongoing rapid increases in GPU performance relative to CPU performance.
The scale of the JUNO detector with many thousands of photomultiplier tubes means 
that most areas of JUNO data processing can naturally benefit from the application 
of GPU parallelism.  Essentually all loops over PMTs are candidates for parallelism, 
meaning that there are opportunities for substantial performance improvements 
across reconstruction, calibration and analysis. 
Profiling application performance should be used to establish priorities, 
I expect there will be many cases of "low hanging fruit" where large benefits 
can be gained simply by straightforward application of standard GPU parallelism techniques.

Although impractical for one person to directly contribute significantly across all areas, 
it is possible to provide examples focussing on one area, and give instruction and 
to make everybody working on JUNO software aware of the benefits and techniques 
of GPU parallelism and to assist them with their development. 
Effective GPU development demands an understanding of CUDA even when 
using the higher level C++ interface provided by Thrust, thus I think it counterproductive
to attempt to add a another layer to hide the details. The best resource to help developers 
are examples of how to structure real world C++ classes that make use of parallelism, 
Opticks has numerous such examples that could be extracted to demonstrate various patterns.

Implementing efficient use of GPU massive parallelism is not an optimization that 
is straightforward to add late in the development of an algorithm, as it 
typically necessitates a total reorganization. Also it is important that developers
start developing their parallelism skills early on while an algorithm is simple.


Event visualization for all ?
-----------------------------------------

Visualization has been an essential tool for the development and debugging of the 
Opticks simulation and has assisted greatly with explaining 
Opticks capabilities to diverse audiences.
The many thousands of PMTs in the JUNO geometry makes it challenging to achieve interactive performance, 
the Opticks OpenGL shader based visualization has reached this level using graphics optimization 
techniques such as instancing and level of detail, when using high performance NVIDIA GPUs.
I would like to make Opticks visualization usable on mainstream hardware, and 
expand the visualizations to provide intuitive views of for example PMT hit times 
and reconstruction algorithm results to assist the understanding and 
development of reconstruction algorithms. 

Performance tests of visualizations of the full and simplified JUNO geometry 
on a variety of hardware are required to assess how best to bring Opticks visualization
to current mainstream hardware.
Opticks can export geometries into the glTF 2.0 3D file format 
which is supported by several existing open source renderers that can run on 
commonly available devices. This provides a way to estimate the kind of performance with the
JUNO geometry that is currently possible prior to implementing a renderer. 
Opticks is structured into ~20 subprojects according to dependencies on 
external packages such as NVIDIA OptiX and CUDA. This design was adopted in order to 
allow most of Opticks to be built on machines with non-NVIDIA GPUs that do not support CUDA.

Opticks provides visualizations of millions of propagating photons that are unprecedented within the field, 
they treat the event time as an input to the render allowing interactive time scrubbing.
High performance visualization is a natural side benefit resulting from my work 
bringing geometry and event data to the GPU, as required for Opticks simulation.


