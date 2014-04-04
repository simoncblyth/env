Overview of Chroma Tests
=========================

This page provides a summary of tests performed using Macbook Pro (late 2013) 
running Mavericks 10.9.1 with NVIDIA GeForce GT 750M GPU. The linked pages
give the gory details.


benchmark
----------

* :doc:`chroma_benchmark`

#. default detector too big for 2GB video memory, switch to `demo.tiny`
#. many uncertainties warnings, fixed by module provided refactor
#. API change caused `pdf_eval` to fail, simple fix 

chroma_cam
------------

* :doc:`chroma_cam`

#. pygame parachute segmentation fault, resolved by not forking the camera 


chroma_sim
-----------

