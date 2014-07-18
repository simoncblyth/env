G4DAE and G4DAEVIEW Development History
=========================================

In August 2013 on rejoining NTU, 
following an extended period focussed on
Dayabay Infrastructure work, I was looking for a
more directly Physics related area of work that would 
suit my experience. I rapidly converged on the perennial 
problem of the slowness of Geant4 optical photon propagation which 
is especially relevant for Dayabay muon simulations. 
Profiling Geant4 simulations confirmed expectations that the 
slowness of tracking literally millions of optical photons through 
detector geometries was the primary cause of the poor performance.

Geant4 is the predominant simulation framework of HEP with a twenty 
year history. It is used by the vast majority of HEP experiments 
both large and small. In addition there are many Geant4 users from the space
science and medical physics communities http://geant4.in2p3.fr/2013/. 
Due to this wide userbase any advances can be hugely useful. 

Surveying prior optimization efforts in this area revealed 
gains of only up to a factor of 10 were expected from  
photon re-weighting techniques although potential for 
significant distortions from the optimization were not excluded.

Following a lead from a BNL colleague I investigated the Chroma
project, which had already implemented on GPU optical photon propagation 
with claims of speedup factors of 200 compared to Geant4.
Chroma achieves these transformational performance gains due to 
a radical change in detector geometry modelling. Rather than  
modelling a detector as a tree of nested solids of particular materials 
Chroma models a detector as a list of oriented triangles 
each representing the boundary between materials. 

Although solid and surface based modelling are conceptually equivalent 
the performance implications of moving to a surface based approach 
are profound due to the overwhelming adoption of surface-based modelling 
in the 3D graphics industry and the past twenty years worth 
of development and optimization of GPUs, the CUDA Toolkit
and 3D graphics libraries such as OpenGL.

I rapidly realized that the drastic approach adopted by Chroma 
was the best way to proceed. However Chroma was developed in the context of prototyping
future detectors using algorithmic detector geometries.
This resulted in Chroma lacking the ability to perform simulations with the 
complex real world geometries of existing detectors.  
The huge potential performance gains however motivated me to embark on 
researching how best to bridge geometry and event data between Geant4 and Chroma.  
After investigating several potential file formats for geometry export including GDML and WRL 
I converged on the industry standard COLLADA 3D asset specification, also known as DAE (Digital Asset Exchange). 
Development of the G4DAE geometry exporter began in October 2013, the exporter 
was structured after the existing Geant4 GDML exporter. It writes DAE files
containing the Geant4 triangulated vertices of the geometry.

Liberating the detector geometry from Geant4 is the first step in bridging to the GPU, 
subsequently a mapping from the G4DAE exported files into Chroma GPU geometries
is needed. Initial technical developments of recreating the Geant4 geometry tree 
and mapping into Chroma geometry proceeded rapidly thanks to the high level of development 
afforded by using open source projects: pycollada, pycuda, numpy and python. 
The bulk of development of the G4DAE exporter was completed by early 2014.

Following purchase of a capabable development machine, a MacBook Pro (2013) with NVIDIA GeForce GT 750M 2048 MB,
in late December 2013 I was finally able to install CUDA and Chroma in mid January 2014.
Getting Chroma to run reliably with this mobile GPU required changes to split up 
the GPU work and to adopt more efficient OpenGL/CUDA interoperation techniques
to improve performance by minimizing transfers between CPU and GPU.    

Up until march 2014 I was visualizing the exported geometry using Chroma raycasting together
with OpenGL viewers from various commercial products and open source 
projects including pycollada and MeshLab. 
Lack of a dedicated tool to work within and poor performance of 
the OpenGL viewers was impeding progress at performing the detailed work of adding 
wavelength dependant material and surface properties. 
In response to this problem in late March I began development of G4DAEVIEW, 
an OpenGL based viewer for G4DAE exported geometries intended to act as a 
backbone application for Chroma testing. The development was based on the 
pyopengl and glumpy open source projects. 

Thanks to excellent examples provided by the glumpy project I was able 
to immediately adopt highly efficient modern OpenGL techniques 
that allowed me to render the entire Dayabay geometry with a single OpenGL draw 
call. The level of performance achieved almost immediately came as an extremely
welcome shock to me. After only a few days of development G4DAEVIEW  
was providing a fluidity of graphics performance that I had never before 
experienced and still frequently leaves me grinning broadly in wonder even now.

Contemplation on this great leap in performance and the relative ease 
with which it was achieved made me come to realize that the underlying causes 
of slow Geant4 track propagation through geometries and the poor performance of 
Geant4/HEP visualizations are one and the same: the Geant4 geometry model. 

Locking the detector geometry data inside a tree of C++ objects in CPU memory prevents 
the benefits of 20 years of GPU development being applied to both HEP visualization 
and track propagation. It forces the adoption of 1990s era OpenGL
rendering techniques that require the transfer of many megabytes of data from CPU to 
GPU for every frame as opposed to modern techniques where the geometry 
data is transferred to the GPU once only, at application launch, and
only a few tens of bytes needs to be transferred for each frame. 

Only after liberating the geometry data can the fruits of billions of dollars and 20 years
of 3D industry development become ripe for the picking, bringing the  
HEP community (as well as Geant4 users from space science and medical physics) 
into the 21st century in terms of track propagation and visualization performance.

The performance bonanza unleashed by G4DAEVIEW enabled me 
to rapidly develop the application into the backbone of 
development and testing of Geant4/Chroma model mapping
and Chroma propagation.
Through the adoption of CUDA/OpenGL interoperation 
techniques it has been straightfoward to implement unprecedented 
(within HEP) interface features such as forward/backward 
time scrubbing through an event propagation.

By the beginning of May 2014 my first pass at transferring the static
geometry data from Geant4 to the GPU with both OpenGL and Chroma/CUDA 
was completed and I turned my attention to runtime integration 
between the Geant4 and Chroma. The requirements are to 
transfer Geant4 generated optical photons to Chroma and 
return the propagated photon results to Geant4 in a manner that 
minimizes code changes at the Geant4 end.  
Fortunately Chroma had already identified the ZeroMQ project as 
a capable and easy to use open source message queue system.
I created ZMQRoot integrating ZeroMQ with ROOT serialization/deserialisation 
and then integrated that with the Dayabay NuWa/Geant4 simulation code. 
By mid May a working system for collecting NuWa/Geant4 optical photons, 
transferring across the network and onto the GPU, propagating and then 
returing the propagated results back to the Geant4 process was operational. 
Although a multi machine approach is convenient during development, 
for best operational performance and reliability a simpler approach 
of colocating the Geant4 and Chroma/CUDA processing the same machine 
can be adopted without major additional development.

At the Beijing IHEP Dayabay Collaboration Meeting at the end of May I 
was able to demonstrate my developments to many colleagues. 
The graphics performance I demonstrated was a surprise and a delight 
to all. 

Following the meeting I worked with an IHEP Juno colleague
on integrating the G4DAE exporter with the Juno simulation code.
We were able to export the Juno geometry and visualize with G4DAEVIEW 
and Chroma Raycasting in under a days work. I demonstrated  
the visualizations of Juno and the Dayabay Chroma propagation to
Juno simulation managers who were sufficiently interested to arrange 
for a GPU desktop machine to be made available to enable further
testing of Chroma, G4DAE and G4DAEVIEW for Juno simulation.

As the G4DAE geometry exporter operates at the Geant4 level it can 
be used by all Geant4 users and it would make most sense to be incorporated
into the Geant4 code base.
The purely geometrical aspects of the export are universally applicable    
and would allow all Geant4 users to visualize their geometries 
with unprecedented performance for example with G4DAEVIEW as well
as commercial COLLADA viewers such as the Preview.app on OSX 
or the open source Blender 3D graphics application on many platforms.

For optical photon simulation with Chroma the details of the material 
and surface properties include detector specific keys that 
need expert mapping to Chroma in order to achieve propagation 
results matching those from Geant4. In addition some aspects 
of detector simulation use detector specific customizations that would 
require careful matching in Chroma.





