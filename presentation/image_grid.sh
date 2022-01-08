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

    export ANNOTATE=1     
    export OUTSTEM=$outstem

    ${IPYTHON:-ipython} -- ~/env/doc/image_grid.py $pathlist
}


./titles.sh 

#make_image_grid "image_grid_overview"  /tmp/urls.txt
make_image_grid "image_grid_cxr_view" /tmp/urls.txt 


