# === func-gen- : tools/hg2git/fastexport fgp tools/hg2git/fastexport.bash fgn fastexport fgh tools/hg2git
fastexport-src(){      echo tools/hg2git/fastexport.bash ; }
fastexport-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fastexport-src)} ; }
fastexport-vi(){       vi $(fastexport-source) ; }
fastexport-env(){      elocal- ; }
fastexport-usage(){ cat << EOU

* https://stackoverflow.com/questions/10710250/converting-mercurial-folder-to-a-git-repository

* http://repo.or.cz/w/fast-export.git

::

    mkdir repo-git # or whatever
    cd repo-git
    git init
    hg-fast-export.sh -r <local-repo>
    git checkout HEAD


::

    Initialized empty Git repository in /Users/blyth/env_git/.git/
    Error: The option core.ignoreCase is set to true in the git
    repository. This will produce empty changesets for renames that just
    change the case of the file name.
    Use --force to skip this check or change the option with
    git config core.ignoreCase false
    delta:env_git blyth$ 



EOU
}
fastexport-dir(){ echo $(local-base)/env/tools/hg2git/fast-export ; }
fastexport-cd(){  cd $(fastexport-dir); }
fastexport-get(){
   local dir=$(dirname $(fastexport-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d fast-export ] && git clone git://repo.or.cz/fast-export.git  
}


fastexport-env-repo()
{
   cd 

   rm -rf env_git
   mkdir env_git
   cd env_git

   git init
   git config core.ignoreCase false

   $(fastexport-dir)/hg-fast-export.sh -r ../env


}
fastexport-env-repo-notes(){ cat << EON

master: Exporting simple delta revision 6303/6307 with 1/3/0 added/changed/removed files
master: Exporting simple delta revision 6304/6307 with 1/4/1 added/changed/removed files
master: Exporting simple delta revision 6305/6307 with 1/8/0 added/changed/removed files
master: Exporting simple delta revision 6306/6307 with 1/4/0 added/changed/removed files
master: Exporting simple delta revision 6307/6307 with 0/2/0 added/changed/removed files
Issued 6307 commands
git-fast-import statistics:
---------------------------------------------------------------------
Alloc'd objects:      65000
Total objects:        64290 (      3572 duplicates                  )
      blobs  :        29870 (      3212 duplicates      14368 deltas of      29706 attempts)
      trees  :        28113 (       360 duplicates      23512 deltas of      26026 attempts)
      commits:         6307 (         0 duplicates          0 deltas of          0 attempts)
      tags   :            0 (         0 duplicates          0 deltas of          0 attempts)
Total branches:           1 (         1 loads     )
      marks:        1048576 (      6307 unique    )
      atoms:           5958
Memory total:          5219 KiB
       pools:          2173 KiB
     objects:          3046 KiB
---------------------------------------------------------------------
pack_report: getpagesize()            =       4096
pack_report: core.packedGitWindowSize = 1073741824
pack_report: core.packedGitLimit      = 8589934592
pack_report: pack_used_ctr            =      33828
pack_report: pack_mmap_calls          =      18444
pack_report: pack_open_windows        =          1 /          1
pack_report: pack_mapped              =   85599855 /   85599855
---------------------------------------------------------------------


delta:~ blyth$ cd env
delta:env blyth$ pyc
=== clui-pyc : in hg/svn/git repo : remove pyc beneath root /Users/blyth/env
delta:env blyth$ cd ..
delta:~ blyth$ diff -r --brief env env_git
Only in env_git: .git
Only in env: .hg
Only in env: _build
diff: env/bin/G4DAEChromaTest.sh: No such file or directory
diff: env_git/bin/G4DAEChromaTest.sh: No such file or directory
diff: env/bin/cfg4.sh: No such file or directory
diff: env_git/bin/cfg4.sh: No such file or directory
diff: env/bin/doctree.py: No such file or directory
diff: env_git/bin/doctree.py: No such file or directory
diff: env/bin/ggv.py: No such file or directory
diff: env_git/bin/ggv.py: No such file or directory
diff: env/bin/ggv.sh: No such file or directory
diff: env_git/bin/ggv.sh: No such file or directory
Only in env/bin: realpath
Only in env/boost/basio: netapp
Only in env/boost/basio: numpyserver
Only in env/boost/basio: udp_server
Only in env/cuda: optix
Only in env/doc: docutils
Only in env/doc: sphinxtest
Files env/env.bash and env_git/env.bash differ
Only in env/graphics/assimp/AssimpTest: build
Only in env/graphics/isosurface: AdaptiveDualContouring
Only in env/graphics/isosurface: ImplicitMesher
Only in env/graphics: oglrap
Only in env/messaging: js
Only in env/network/asiozmq: examples
Only in env/network/gputest: .gputest.bash.swp
Only in env/npy: quartic
Only in env/numerics: npy
Only in env_git/nuwa/MockNuWa: MockNuWa.cc
Only in env/nuwa/MockNuWa: mocknuwa.cc
Only in env: ok
Only in env/optix/lxe: .cfg4.bash.swp
Only in env/presentation: intro_to_cuda
Only in env/tools: hg2git
delta:~ blyth$ 


delta:~ blyth$ diff env/env.bash env_git/env.bash
1938d1937
< fastexport-(){      . $(env-home)/tools/hg2git/fastexport.bash && fastexport-env $* ; }
delta:~ blyth$ 


delta:env_git blyth$ git shortlog -e -s -n
   
   ..list of email addesses and commit counts
   ..observe lots of near dupe email addresses
   ..mapping mapping can cleanup

elta:env_git blyth$ 

delta:~ blyth$ du -hs env env_git
125M    env
127M    env_git



EON
}



