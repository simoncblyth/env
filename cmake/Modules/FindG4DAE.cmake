
set(G4DAE_PREFIX "$ENV{LOCAL_BASE}/env/g4d")

find_library( G4DAE_LIBRARIES 
              NAMES G4DAE
              PATHS ${G4DAE_PREFIX}/lib )

set(G4DAE_INCLUDE_DIRS "${G4DAE_PREFIX}/include")
set(G4DAE_DEFINITIONS "")

