#!/bin/bash -l 

usage(){ cat << EOU

Without a second path argument the output html is written to stdout

EOU
}

txt=${1:-opticks_20211223_pre_xmas.txt}

#TITLEMATCH=overview ipython --pdb -- titles.py $txt /tmp/out.html

TITLEMATCH=cxr_view_cam_0 ipython --pdb -- titles.py $txt /tmp/out.html




