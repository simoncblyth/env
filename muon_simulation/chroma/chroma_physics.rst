Chroma Physics
================

Is the modeling equivalent to Geant4 ?
-----------------------------------------

Photon enum::

     47 enum
     48 {
     49     NO_HIT           = 0x1 << 0,
     50     BULK_ABSORB      = 0x1 << 1,
     51     SURFACE_DETECT   = 0x1 << 2,
     52     SURFACE_ABSORB   = 0x1 << 3,
     53     RAYLEIGH_SCATTER = 0x1 << 4,
     54     REFLECT_DIFFUSE  = 0x1 << 5,
     55     REFLECT_SPECULAR = 0x1 << 6,
     56     SURFACE_REEMIT   = 0x1 << 7,
     57     SURFACE_TRANSMIT = 0x1 << 8,
     58     BULK_REEMIT      = 0x1 << 9,
     59     NAN_ABORT        = 0x1 << 31
     60 }; // processes



