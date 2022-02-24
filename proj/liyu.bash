liyu-source(){   echo ${BASH_SOURCE} ; }
liyu-edir(){ echo $(dirname $(liyu-source)) ; }
liyu-ecd(){  cd $(liyu-edir); }

liyu-repodir(){  echo $HOME/liyu ; }

liyu-dir(){      echo $(liyu-repodir)/Rich_Simplified/src ; }
liyu-cd(){   cd $(liyu-dir); }

liyu-vi(){   vi $(liyu-source) ; }
liyu-env(){  elocal- ; }
liyu-usage(){ cat << EOU



Dear Simon,

During the recent meeting to prepare for the Hackathon, you mentioned about
getting a copy of the simple geometry that we hope to test during the
Hackathon. As you mentioned this would help in preparing for the Hackathon.

It is available at the following link, which you would be able to access:

https://gitlab.cern.ch/liyu/opticks/-/tree/master/Rich_Simplified

In this page, the ‘include’ directory contains the header files, some of
which have the actual geometry parameters. The ‘src’ contains the code, which
is mostly similar to those in standard Geant4 examples. In particular the file
RichTbOpticksDetectorConstruction.cc has the Construct() method which creates
the full geometry. The RichTbGraphicsLbR.cc contains the graphics settings for
standard Geant4 visualization.

The main page of the link has a README file with some instructions on how to install this.

The geometry contains two spherical mirrors. One of them is ‘almost flat’
with a large radius of curvature. The geometry also has a quartz window and an
array of Mapmts arranged inside some structures. We also attach couple of
pictures of this geometry for illustration.

Hope this is sufficient information. If you have any questions related to this, please let us know.

BTW, thanks a lot for fixing issues related to conversion of spherical segments
into Opticks, as mentioned in your recent message to us on February 15. In our
case it was the ‘almost flat’ spherical mirror segment mentioned above, which
was causing the issues in this context. Whenever these fixes are ready and
available, please let us know so that we can try running Opticks with this
geometry.

Thanks and regards,

   Sajan, Adam, Keith, Yunlong, Lucas,Evelina and Marco. 



epsilon:~ blyth$ git clone https://gitlab.cern.ch/liyu/opticks.git Rich_Simplified

Use Github web interface to creaty empty repo called "Rich_Simplified"

epsilon:Rich_Simplified blyth$ git remote add github git@github.com:simoncblyth/Rich_Simplified.git
epsilon:Rich_Simplified blyth$ git branch -M main

      -M
           Shortcut for --move --force.

      -m, --move
           Move/rename a branch and the corresponding reflog.

      -f, --force
           Reset <branchname> to <startpoint> if <branchname> exists already. Without -fgit branch refuses to change an existing branch. In combination with -d (or --delete), allow deleting the branch
           irrespective of its merged status. In combination with -m (or --move), allow renaming the branch even if the new branch name already exists.


epsilon:Rich_Simplified blyth$ git push -u github main

      -u, --set-upstream
           For every branch that is up to date or successfully pushed, add upstream (tracking) reference, used by argument-less git-pull(1) and other commands. For more information, see
           branch.<name>.merge in git-config(1).



Compare and copy over meaningful changes

epsilon:~ blyth$ diff -r -- brief liyu Rich_Simplified
diff: extra operand `Rich_Simplified'
diff: Try `diff --help' for more information.
epsilon:~ blyth$ diff -r --brief liyu Rich_Simplified
Files liyu/.git/HEAD and Rich_Simplified/.git/HEAD differ
Files liyu/.git/config and Rich_Simplified/.git/config differ
Files liyu/.git/index and Rich_Simplified/.git/index differ
Files liyu/.git/logs/HEAD and Rich_Simplified/.git/logs/HEAD differ
Only in Rich_Simplified/.git/logs/refs/heads: main
Only in liyu/.git/logs/refs/heads: master
Only in Rich_Simplified/.git/logs/refs/remotes: github
Files liyu/.git/logs/refs/remotes/origin/HEAD and Rich_Simplified/.git/logs/refs/remotes/origin/HEAD differ
Only in Rich_Simplified/.git/refs/heads: main
Only in liyu/.git/refs/heads: master
Only in Rich_Simplified/.git/refs/remotes: github


Files liyu/Rich_Simplified/CMakeLists.txt and Rich_Simplified/Rich_Simplified/CMakeLists.txt differ

    cp liyu/Rich_Simplified/CMakeLists.txt Rich_Simplified/Rich_Simplified/CMakeLists.txt

Files liyu/Rich_Simplified/Rich_Simplified.cc and Rich_Simplified/Rich_Simplified/Rich_Simplified.cc differ

     cp liyu/Rich_Simplified/Rich_Simplified.cc Rich_Simplified/Rich_Simplified/Rich_Simplified.cc 

Files liyu/Rich_Simplified/TimeTest.cc and Rich_Simplified/Rich_Simplified/TimeTest.cc differ

     cp liyu/Rich_Simplified/TimeTest.cc  Rich_Simplified/Rich_Simplified/TimeTest.cc

Only in liyu/Rich_Simplified: build
Files liyu/Rich_Simplified/include/RichTbSimH.hh and Rich_Simplified/Rich_Simplified/include/RichTbSimH.hh differ


    diff liyu/Rich_Simplified/include/RichTbSimH.hh Rich_Simplified/Rich_Simplified/include/RichTbSimH.hh 

Files liyu/Rich_Simplified/src/EventAction.cc and Rich_Simplified/Rich_Simplified/src/EventAction.cc differ

    diff liyu/Rich_Simplified/src/EventAction.cc Rich_Simplified/Rich_Simplified/src/EventAction.cc  


Files liyu/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc and Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc differ

    diff liyu/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1FlatMirror.cc 

Files liyu/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc and Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc differ

    diff liyu/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc Rich_Simplified/Rich_Simplified/src/RichTbLHCbR1SphMirror.cc 


Files liyu/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc and Rich_Simplified/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc differ

   diff liyu/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc Rich_Simplified/Rich_Simplified/src/RichTbOpticksDetectorConstruction.cc 


Files liyu/Rich_Simplified/src/RichTbSimH.cc and Rich_Simplified/Rich_Simplified/src/RichTbSimH.cc differ

   diff liyu/Rich_Simplified/src/RichTbSimH.cc Rich_Simplified/Rich_Simplified/src/RichTbSimH.cc 


Only in Rich_Simplified/Rich_Simplified/src: RichTbVisManager.cc
Only in liyu/Rich_Simplified/src: RichTbVisManager.cc_

   

Files liyu/Rich_Simplified/src/SensitiveDetector.cc and Rich_Simplified/Rich_Simplified/src/SensitiveDetector.cc differ

   diff liyu/Rich_Simplified/src/SensitiveDetector.cc Rich_Simplified/Rich_Simplified/src/SensitiveDetector.cc 

Only in liyu/Rich_Simplified/src: load.sh
Only in liyu/Rich_Simplified/src: tt.sh
epsilon:~ blyth$ 
epsilon:~ blyth$ 




EOU
}


liyu-get(){
   local dir=$(dirname $(liyu-repodir)) &&  mkdir -p $dir && cd $dir
   git clone https://gitlab.cern.ch/liyu/opticks.git
}
