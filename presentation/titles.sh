#!/bin/bash -l 

usage(){ cat << EOU

Without a second path argument the output html is written to stdout

EOU
}


#default=opticks_20211223_pre_xmas.txt
default=opticks_autumn_20211019.txt

txt=${1:-$default}
make_image_urls_list()
{ 
   ipython --pdb -- titles.py $1 /tmp/out.html ; 
}

if [ "$txt" == "opticks_20211223_pre_xmas.txt" ]; then 

    #TITLEMATCH=overview make_image_urls_list $txt
    TITLEMATCH=cxr_view_cam_0 make_image_urls_list $txt

elif [ "$txt" == "opticks_autumn_20211019.txt" ]; then

    #TITLEMATCH=QCKTest_1  make_image_urls_list $txt
    TITLEMATCH=qcktest_1  make_image_urls_list $txt

fi 







