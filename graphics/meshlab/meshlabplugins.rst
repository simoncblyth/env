Meshlab Plugins 
=================

Collada Import
----------------

Meshlab collada importer takes 40 minutes to import g4_00.dae, compared to 40s by pycollada, 
suspect horrendously inefficient XML handling is the culprit.

Despite collada import functionality coming mainly from VCGLIB, there is 
enough meshlab (eg the Meshmodel) in the importer to make development
within meshlab to be more appropriate that attempting to operate 
at vcglib level.


Investigation environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:meshlab blyth$ pwd
    /Users/blyth/e/graphics/meshlab
    simon:meshlab blyth$ cp $(meshlab-dir)/meshlabplugins/io_collada/io_collada.pro meshlabplugins/io_collada/




