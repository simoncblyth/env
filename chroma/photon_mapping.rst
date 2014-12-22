Photon Mapping
===============

A technique used in ray tracers to provide global illumination
based on density of photon hits 
might provide an interesting way to "see" what events 
looks like inside the ADs.


Accelerated Stochastic Progressive Photon Mapping On GPU
----------------------------------------------------------

* https://github.com/ishaan13/PhotonMapper
* https://github.com/ishaan13/PhotonMapper/blob/master/PROJ1_NIX/makefile

  * fairly minimal dependencies



Opposite Renderer
-------------------

* http://apartridge.github.io/OppositeRenderer/
* https://github.com/igui/OppositeRenderer

This project is a part of Stian Pedersen's master's thesis project at NTNU.
This demo renderer contains a GPU implementation of the Progressive Photon
Mapping algorithm. It is written in C++ using CUDA and OptiX. The renderer has
a GUI and can load scenes from the well known Collada (.dae) scene format. It
has a client-server architecture allowing multi-GPU (distributed) rendering
with any number of GPUs. It has been tested with up to six GPUs via Gigabit
ethernet with very good speedup.


MNRT
-----

* http://www.maneumann.com/MNRT/


Mathias Neumann:

    GPU-based global illumination using CUDA MNRT is a demo application I developed
    while working on my Diplomarbeit (German master thesis equivalent) about
    GPU-based global illumination. This page is a collection of links to content
    about MNRT.


 
