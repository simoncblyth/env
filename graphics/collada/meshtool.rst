Meshtool
==========

Meshtool provides PyCollada based utilities such as:

#. viewing collada docs (via Panda3D)
#. dumping info on collada docs

Usage Examples
----------------

::

    simon:~ blyth$ meshtool-
    simon:~ blyth$ t meshtool
    meshtool is a function
    meshtool () 
    { 
        /usr/bin/python -c "from meshtool.__main__ import main ; main() " $*
    }

    simon:~ blyth$ cd /usr/local/env/graphics/collada
    simon:collada blyth$ curl -O http://localhost:8080/subcopy/__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0.dae
    simon:collada blyth$ meshtool --load_collada __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0.dae --viewer
 
          ## collada_viewer shows nothing visible, as have no cameras/lights in the file

    simon:~ blyth$ meshtool --load_collada http://localhost:8080/subcopy/3199.dae --viewer



To edit the DAE download first::

    simon:~ blyth$ curl -O http://localhost:8080/subcopy/3199.dae
    simon:~ blyth$ meshtool --load_collada 3199.dae --viewer



