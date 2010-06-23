#!/bin/bash
#
# Environment isolation shim ...
#
#   Usage : 
#       cd NuWa-trunk
#       path/to/isotest.sh rootiotest 
#       $(env-home)/offline/isotest.sh rootiotest 
#
#  see runtest.sh for details
#
exec env -i $(dirname $0)/runtest.sh $*

