#!/bin/bash -l

chroma-
which python


#
# Latest DAE, exported on N and copied elsewhere as g4_00.dae.6 now g4_00.dae 
#
#      $LOCAL_BASE/env/geant4/geometry/export/DVGX_20140222-1423/g4_00.dae
#      $ENV_HOME/geant4/geometry/materials/g4_00.dae
# 
#

dae=$ENV_HOME/geant4/geometry/materials/g4_00.dae
ls -l $dae
du -hs $dae


./demo_collada_to_chroma.py $dae

