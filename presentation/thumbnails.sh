#!/bin/bash -l 

usage(){ cat << EOU
thumbnails.sh
===============



EOU
}


${IPYTHON:-ipython} ./thumbnails.py
