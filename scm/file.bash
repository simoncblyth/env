
#
#  put/get to same place in file heirarchy on remote node, assuming it exists
#
#   file-rpwd
#   file-p
#   file-g
#
#
#   idea ...  have LOCAL_BASE for all nodes , so can create a LOCAL_BASE relative copy 
#
#

file-rpwd(){     echo $PWD | perl -p -e 's|$ENV{"HOME"}/(.*)|$1|' ;  } ## returns the path relative to home
file-p(){ [ -f "$1" ] && scp $1  ${2:-$TARGET_TAG}:$(file-rpwd)/  || echo need at least one argument ; }
file-g(){ [ -z "$1" ] || scp ${2:-$TARGET_TAG}:$(file-rpwd)/$1 .  || echo need at least one argument ; }




