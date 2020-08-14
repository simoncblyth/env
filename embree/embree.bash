# === func-gen- : embree/embree fgp embree/embree.bash fgn embree fgh embree
embree-src(){      echo embree/embree.bash ; }
embree-source(){   echo ${BASH_SOURCE:-$(env-home)/$(embree-src)} ; }
embree-vi(){       vi $(embree-source) ; }
embree-env(){      elocal- ; }
embree-usage(){ cat << EOU

Embree : High Performance Ray Tracing Kernels
================================================

* https://embree.github.io



OSPRay 
---------

See also 

* https://www.ospray.org

OSPRay internally builds on top of Intel Embree and ISPC (Intel SPMD Program
Compiler), and fully exploits modern instruction sets like Intel SSE4, AVX,
AVX2, and AVX-512 to achieve high rendering performance, thus a CPU with
support for at least SSE4.1 is required to run OSPRay.

* https://tacc.github.io/vtkOSPRay/

vtkOSPRay is a VTK module which utilizes Intel's OSPRay ray tracing framework
(http://ospray.github.io) for rendering. This offers a performant CPU rendering
package with enhanced image quality. This library is the base for OSPRay
plugins for the commonly used visualization tools ParaView and VisIt.



Refs
------

* https://www.embree.org
* https://www.embree.org/data/embree-siggraph-2018-final.pdf
* https://embree.github.io/data/embree-siggraph-2016-final.pdf

* https://embree.github.io/downloads.html


The kernels are optimized for photo-realistic rendering on the latest Intel® processors 
with support for SSE, AVX, AVX2, and AVX512


Seems github not allowing downloads with curl ?

* https://medium.com/@caludio/how-to-download-large-files-from-github-4863a2dbba3b#.3desirl2b

Turned out to be pilot error, there is a good reason to leave the URL as copy and pasted
in and not do replacememt...


::

    simon:doc blyth$ embree-src-get
    Cloning into 'embree'...
    remote: Counting objects: 90238, done.
    remote: Total 90238 (delta 0), reused 0 (delta 0), pack-reused 90238
    Receiving objects: 100% (90238/90238), 35.31 MiB | 1.82 MiB/s, done.
    Resolving deltas: 100% (75049/75049), done.
    Checking connectivity... done.
    Note: checking out 'v2.13.0'.

    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by performing another checkout.

    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -b with the checkout command again. Example:

      git checkout -b new_branch_name

    HEAD is now at 4714fd1... compile fix for ancient tbb versions without task_arean support
    simon:embree blyth$ 



EOU
}


embree-url(){ echo https://github.com/embree/embree/releases/download/v2.13.0/embree-2.13.0.x86_64.macosx.tar.gz ; }
embree-dst(){ echo $(basename $(embree-url)) ; }

#embree-nub(){ local dst=$(embree-dst) ; echo ${dst/.tar.gz} ; } 
embree-nub(){  echo embree; } 


embree-sub(){ echo $(basename $(dirname $(embree-url))) ; }   # eg v2.13.0
embree-dir(){ echo $(local-base)/env/embree/$(embree-nub) ; }
embree-cd(){  cd $(embree-dir); }

embree-dist-get(){
   local dir=$(dirname $(embree-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(embree-url)
   local dst=$(embree-dst)
   [ ! -f "$dst" ] && curl $url -L -o $dst

   local nub=$(embree-nub)
   [ ! -d $nub ] && tar zvxf $dst  

}

embree-src-get(){

  local dir=$(dirname $(embree-dir)) &&  mkdir -p $dir && cd $dir

  local nub=$(embree-nub)
  [ "$nub" != "embree" ] && echo unexpected nub for src $nub && return 

  [ -d "$nub" ] && return 
  
  git clone https://github.com/embree/embree.git $nub
  cd $nub
  git checkout v2.13.0

}

