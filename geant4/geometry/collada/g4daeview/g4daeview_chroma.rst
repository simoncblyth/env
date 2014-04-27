G4DAEView Chroma
===================

Chroma raycast speed depends primarily on geometrical complexity.
For fast testing, use partial geometries.


Single PMT Test
-----------------

::

   g4daeview.sh -g 6000 --with-chroma --launch 1,1,1 --near 1


A Few PMTs and radial shield
-----------------------------

::

   g4daeview.sh -g 6000:6100 --with-chroma --launch 1,1,1 --near 1



