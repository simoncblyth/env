EventDisplay Coordinates
==========================

Users specify which local transform to apply to global 
coordinates (which frame is the global one, really universe ?) 
by means of DE (detector element) names starting `/dd/Structure/...`

In order to make sense of any such coordinates
(eg to display ntuple/histogram content within the 3D
representation) need to have access to the G4AffineTransforms
for all DE.  Currently the IDMAP machinery 
only caches transforms for cathode volumes with PmtId attached.  

TODO:

#. during geometry traverse already constructing G4TouchableHistory 
   for every volumes 

#. harvest G4AffineTransform for all DE obtained with TH2DE 
   into a persistable map keyed on DE name, similar to G4DAETransformCache
   perhaps G4DAETransformMap 


All DE are traversed by::

    [blyth@ntugrid5 ~]$ which de.sh
    ~/env/bin/de.sh



