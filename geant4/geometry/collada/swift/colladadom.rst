Apples SceneKit uses ColladaDOM it seems
==========================================

Many warnings from meta, extra, bordersurface, opticalsurface::

    ColladaDOM Warning: The DOM was unable to create an element named meta at line 129650. Probably a schema violation.

Strip the extra elements::

    xsltproc $ENV_HOME/geant4/geometry/collada/swift/strip-extra-meta.xsl /tmp/g4_00.dae > /tmp/g4_00_stripped.dae


