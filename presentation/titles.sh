#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

usage(){ cat << EOU

Without a second path argument the output html is written to stdout

EOU
}


presentation-


#default=opticks_20211223_pre_xmas.txt
#default=opticks_autumn_20211019.txt
#default=opticks_20220115_innovation_in_hep_workshop_hongkong.txt
default=$(presentation-iname).txt

txt=${1:-$default}

if [ ! -f "$txt" ]; then
   echo $msg there is no presentation txt $txt 
   exit 1 
fi 


#export PREFIX=https://simoncblyth.bitbucket.io
export PREFIX=""  # have move to using relative urls so html works on three servers unchanged

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


else
    make_image_urls_list $txt
fi 


cmd="cat /tmp/urls.txt"
echo $msg $cmd
eval $cmd

exit 0 

