#!/bin/bash -l 

usage(){ cat << EOU

1. use titles.sh to extract lists of image paths used as s5_background_image in s5 presentations  
   which are written to /tmp/urls.txt

2. use this script to create composite image grids containing all the listed images


::

    HEADGAP=1 ./image_grid.sh 

EOU
}


make_image_grid()
{
    local outstem=$1
    local pathlist=$2
    local annolist=$3

    export ANNOTATE=1     
    export OUTSTEM=$outstem

    ${IPYTHON:-ipython} -- ~/env/doc/image_grid.py $pathlist $annolist
}


#./titles.sh 

expts(){ cat << EOL
/env/graphics/ggeoview/jpmt-inside-wide_half.png
/env/presentation/dayabay-principal_half.png
/env/presentation/LHCb_RICH/OKTest_rich1_new.png
/env/presentation/LZ/LZ_with_Opticks_half.png
EOL
}
anno(){ cat << EOL
JUNO
Dayabay
LHCb RICH
   LZ
EOL
}

expts > /tmp/urls.txt
anno  > /tmp/anno.txt

#make_image_grid "image_grid_overview"  /tmp/urls.txt
#make_image_grid "image_grid_cxr_view" /tmp/urls.txt 

make_image_grid "image_grid_opticks_generality" /tmp/urls.txt /tmp/anno.txt

