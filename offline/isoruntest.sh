#!/bin/bash
#
# Enforced isolation shim ...
#
#   Usage : 
#       cd path/to/NuWa-trunk
#       BUILD_PATH=dybgaudi/trunk/RootIO/RootIOTest $(env-home)/offline/isoruntest.sh 
#   OR :
#       $(env-home)/offline/isoruntest.sh dybgaudi/trunk/RootIO/RootIOTest 
#
#   Hmm add such a function to dybinst to allow...
#       ./dybinst trunk test dybgaudi/trunk/RootIO/RootIOTest     
#
#       ./dybinst trunk test RootIOTest      ... can use the cmt ROOIOTESTROOT envvar to get the dir and thence the path 
#       ./dybinst trunk test 
#
#
exec env -i BUILD_PATH=${1:-$BUILD_PATH} $(dirname $0)/runtest.sh

