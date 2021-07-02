#!/bin/bash -l 

notes(){ cat << EON

DONT DO THIS ANYMORE 

UNLIKE SLIDE RST SOURCES THE JAVASCRIPT AND CSS ARE NOW BEING 
DIRECTLY MANAGED IN THE PRESENTATION REPO, NOT MANAGED
HERE AND COPIED ACROSS

MAKE CHANGES AND DIRECTLY SEE THE EFFECTS BY 
EDITING IN THE BELOW DIR:

   ~/simoncblyth.bitbucket.io/env/presentation/ui/my-small-white 

EON
}

notes

exit 1


srcdir=$HOME/env/presentation/ui/my-small-white
dstdir=$HOME/simoncblyth.bitbucket.io/env/presentation/ui/my-small-white

names="pretty.css slides.js"

for name in $names ; do 
   src=$srcdir/$name
   dst=$dstdir/$name
   diff $src $dst 
   #cmd="cp $src $dst"
   #echo $cmd
   #eval $cmd
done
 


