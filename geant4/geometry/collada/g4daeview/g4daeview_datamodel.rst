Datamodel
==========

Organize the plethora of quantities into entities.

Classification considerations.

#. parameters for which changes demand restart of the worker

   * geometry file
   * geometry selection
   * flavor of the worker, non-vbo OR vbo
   * propagation kernel 

#. parameters of kernel launch, easily tuned

   * threads per block
   * max blocks, determines how the work gets split into kernel launches 
   
#. result timings 

   * investigate scaling with number of photons to see 
     what makes sense for comparing different events 







