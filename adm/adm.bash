adm-src(){      echo adm/adm.bash ; }
adm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(adm-src)} ; }
adm-vi(){       vi $(adm-source) ; }
adm-usage(){ cat << EOU

ADM : Python Virtualenv for SysAdmin 
======================================

Overview
----------

Bash wrapper for creating and using the **ADM** 
python virtualenv.

Scope of ADM virtualenv
--------------------------

House python packages needed for sysadmin tasks that do not need 
to be generally available in the system or macports pythons.  For
example:

#. hgapi, programmatic access to Mercurial repository 


Future functionality : convert hg to git 
------------------------------------------

* http://arr.gr/blog/2011/10/bitbucket-converting-hg-repositories-to-git/


FUNCTIONS
-----------

*adm-svnmirror-init name*

     Setup SVN mirror repository connected to source repository via
     configured src repos URL eg http://dayabay.phys.ntu.edy.tw/repos
     OR envvar ADM_SVNREPOSURL 

*adm-svnmirror-sync name*

     sync SVN mirror repository 

*adm-convert*

     Runs hg convert, migrating SVN repo to Mercurial repo 

     adm-convert env
     adm-convert heprez
     adm-convert tracdev

     Conversion of about 4000 env revisions from local file based SVN 
     repo (updated via adm-svnmirror-sync) takes about 1 minute.

*adm-prep-hg name*


*adm-compare-svnhg name*

     Compares the SVN and converted HG repositories 
     by log comparison with timestamp matching to map between revisions.
     Makes corresponding SVN and HG checkouts for every revision, 
     compares file paths and content digests for the SVN and HG working copy.

*adm-authormap*

     from SVN username into Mercurial/Bitbucket name/email


*adm-utilities*

     Installs basic utilities: eg readline, ipython 



Switching svnsync mirror repo source URL following network rejig
------------------------------------------------------------------

* http://www.emreakkas.com/linux-tips/how-to-change-svnsync-url-for-source-repository





Setup local SVN mirror for faster SVN to HG conversions
---------------------------------------------------------

::

    (adm_env)delta:~ blyth$ adm-
    (adm_env)delta:~ blyth$ adm-svnmirror-init heprez
    svnsync init file:///var/scm/subversion/heprez http://dayabay.phys.ntu.edu.tw/repos/heprez
    Copied properties for revision 0.

    (adm_env)delta:~ blyth$ adm-svnmirror-sync heprez
    svnsync sync file:///var/scm/subversion/heprez
    Committed revision 1.
    Copied properties for revision 1.
    Transmitting file data ..............................
    Committed revision 2.
    Copied properties for revision 2.
    Transmitting file data .
    ...
    Copied properties for revision 955.
    Transmitting file data .
    Committed revision 956.
    Copied properties for revision 956.
    (adm_env)delta:~ blyth$ 


Simple check::

    (adm_env)delta:~ blyth$ svn co file:///var/scm/subversion/heprez/trunk heprez_svn
    (adm_env)delta:~ blyth$ svn co http://dayabay.phys.ntu.edu.tw/repos/heprez/trunk heprez_rsvn
    (adm_env)delta:~ blyth$ diff -r --brief heprez_svn heprez_rsvn
    Files heprez_svn/.svn/wc.db and heprez_rsvn/.svn/wc.db differ
    diff: heprez_svn/sources/belle/orig: No such file or directory
    diff: heprez_rsvn/sources/belle/orig: No such file or directory
    # warning due a broken link in both cases,  orig -> /Users/blyth/hf/sources/belle



Perform conversion::

    (adm_env)delta:~ blyth$ adm-
    (adm_env)delta:~ blyth$ rm -rf /var/scm/mercurial/heprez
    (adm_env)delta:~ blyth$ adm-convert heprez

    === adm-convert : filemap /Users/blyth/.heprez/filemap.cfg
    #placeholder

    === adm-convert : authormap /Users/blyth/.heprez/authormap.cfg
    ...

    hg convert --config convert.localtimezone=true --source-type svn --dest-type hg file:////var/scm/subversion/heprez /var/scm/mercurial/heprez --filemap /Users/blyth/.heprez/filemap.cfg --authormap /Users/blyth/.heprez/authormap.cfg
    === adm-convert : enter YES to proceed YES
    scanning source...
    sorting...
    converting...


Tracdev
--------

::

    rm -rf /var/scm/mercurial/tracdev
    adm-convert tracdev


Related
--------

#. *hg-*
#. *scmmigrate-*
#. *hgapi-*


Issues
-------

svn bindings access 
~~~~~~~~~~~~~~~~~~~~~~

Need to manually arrange access to SVN bindings, how is that done for chroma_env ?::

    sys.path.append('/opt/local/lib/svn-python2.7')
    import svn 

For a more permanent workaround use *adm-svn-bindings*.
Its unclear why that is needed. The macports pkg contains the pth but that 
seems not to get propagated via virtualenv::

    delta:~ blyth$ port contents subversion-python27bindings
    Port subversion-python27bindings contains:
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/svn-python.pth


main site-packages access (July 30, 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Motivated by pysvn access, applied adm-site-packages 


env access
~~~~~~~~~~~

Also *adm-env-ln*


status Jul 30, 2014
~~~~~~~~~~~~~~~~~~~~~

#. env is standard svn layout with only trunk populated

   * converted to hg and 1st pass verified, some tricky areas of SVN history 
     needed workarounds 
   * TODO: more verification, check with/without trunk pros/cons

#. heprez is standard layout with only trunk populated

   * converted to hg 

#. tracdev has multiple trunk/branches/tags under multiple toplevel names, will need some special filemappings ?
   
   * http://dayabay.phys.ntu.edu.tw/repos/tracdev/ 


EOU
}
adm-env(){      
   elocal- ; 
   adm-activate
}
adm-activate(){
   local dir=$(adm-dir)
   [ -f "$dir/bin/activate" ] && source $dir/bin/activate 
}
adm-dir(){ echo $(local-base)/env/adm_env ; }
adm-sitedir(){ echo $(adm-dir)/lib/python2.7/site-packages ; }
adm-sitedir-cd(){ cd $(adm-sitedir) ; }
adm-cd(){  cd $(adm-dir); }
adm-mate(){ mate $(adm-dir) ; }
adm-get(){
   local dir=$(dirname $(adm-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(basename $(adm-dir))
   [ ! -d "$nam" ] && echo $msg CREATING VIRTUALENV $dir && virtualenv $nam
}
adm-info(){
   which python
   which pip
   which easy_install

   python -c "import sys ; print '\n'.join(sys.path) "
}

adm-assert(){
   local msg="=== $FUNCNAME :"
   [ -z "$VIRTUAL_ENV" ] && echo $msg requires VIRTUAL_ENV && sleep 100000000
   [ "$(basename $VIRTUAL_ENV)" != "adm_env" ] && echo $msg NOT IN ADM ENV DO adm- FIRST && sleep 100000000
}

adm-utilities(){
   adm-assert 

   easy_install readline   # see ipython- notes
   pip -v install ipython
}

adm-env-ln(){ 
   ln -s $(env-home) $(adm-sitedir)/env
}
adm-svn-bindings(){
   echo /opt/local/lib/svn-python2.7 > $(adm-sitedir)/svn-python.pth 
}
adm-site-packages(){
   # potential to open a can of worms, but I want pysvn installed from macports access 
   echo /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages > $(adm-sitedir)/site-packages.pth
}



adm-svnrepodbdir(){
  case $1 in 
    env) echo /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 ;; 
  esac
}


adm-svnurl(){
  local repo=$1
  case $repo in 
    env_remote) echo http://dayabay.phys.ntu.edu.tw/repos/$repo/trunk ;;    
    env)     echo file:///var/scm/subversion/env/trunk ;;     # NB no actual trunk directory in file system, its a fiction understood to be inside the repo database   
    heprez)  echo file:///var/scm/subversion/heprez/trunk ;;
    tracdev) echo file:///var/scm/subversion/tracdev ;;      # no trunk
          *) echo file:///var/scm/subversion/$repo/trunk ;;  # default to SVN standard layout, NB no trunk in filesystem, it lives inside the repo DB
  esac
}

adm-svnreposurl(){ echo ${ADM_SVNREPOSURL:-http://dayabay.phys.ntu.edy.tw/repos} ; }
adm-svnmirror-init(){
    local name=$1
    local fold=/var/scm/subversion
    mkdir -p $fold

    local repo=$fold/$name
    [ -d $repo ] &&  echo $msg repo $repo exists already && return 

    [ ! -d $repo ] &&  svnadmin create $repo
    echo '#!/bin/sh' > $repo/hooks/pre-revprop-change
    chmod +x $repo/hooks/pre-revprop-change

    local cmd="svnsync init file://$repo $(adm-svnreposurl)/$name"
    echo $cmd
    eval $cmd 
}

adm-svnmirror-get-url(){
    local name=${1:-env}
    local fold=/var/scm/subversion
    local repo=$fold/$name
    svn propget svn:sync-from-url --revprop -r 0 file://$repo
}

adm-svnmirror-set-url(){
    local name=${1:-env}
    local url=${2:-you-need-to-provide-new-url}
    local fold=/var/scm/subversion
    local repo=$fold/$name
    local cmd="svn propset svn:sync-from-url --revprop -r 0 $url file://$repo"
    echo $msg $cmd 
} 


adm-svnmirror-get-uuid(){
    local name=${1:-env}
    local fold=/var/scm/subversion
    local repo=$fold/$name
    svn propget svn:sync-from-uuid --revprop -r 0 file://$repo
}




adm-svnmirror-sync(){
    local name=${1:-env}
    local fold=/var/scm/subversion
    mkdir -p $fold
    local repo=$fold/$name
    local cmd="svnsync sync file://$repo"
    echo $cmd
    eval $cmd
}


adm-hgsvnrev(){
  ## SVN rev 1 presumably creates the trunk folder, this presumably aligns history to trunkless hg  
  case $1 in 
    heprez)  echo 0 1 ;;
    tracdev) echo 0 1 ;;
    env) echo 1583 1596 ;;
    env0) echo 1600 1598 ;;
    env1) echo 3470 1598 ;;
    env2) echo 0 1 ;;
    env3) echo 724 732 ;;
 workflow0) echo 679 691 ;;        
       *) echo 0 1 ;;        
  esac
}

adm-opts(){
  case $1 in 
        env) echo --skipempty --ignore-externals --clean-checkout-revs 1599,1600,1601 --known-bad-revs 1600 ;;
     heprez) echo --skipempty --ignore-externals ;;
    tracdev) echo --skipempty --ignore-externals --expected-svnonly-dirs xsltmacro/branches ;;
   workflow) echo --skipempty --ignore-externals --known-bad-paths notes/dev/tools/subversion.rst ;;
   # r696: notes/dev/tools/subversion.rst uses SVN variable substitution macros, so mismatch is expected
          *) echo --skipempty --ignore-externals ;;
  esac 
}

adm-prep-hg(){

   local name=${1:-env}
   local vard=/var/scm/mercurial/$name
   #local repd=$HOME/$name
   local tmpd=/tmp/mercurial/$name
   mkdir -p $(dirname $tmpd)

   [ ! -d "$vard" ] && echo $msg ERROR no $vard && return
   #[ ! -d "$repd" ] && hg --cwd $(dirname $repd) clone $vard 
   [ ! -d "$tmpd" ] && hg --cwd $(dirname $tmpd) clone $vard 
}


adm-prep-svn(){

   local name=${1:-env}
   local vard=/var/scm/subversion/$name
   local repd=$HOME/${name}_svn
   local tmpd=/tmp/subversion/$name

   [ ! -d "$vard" ] && echo $msg ERROR no $vard && return
   [ ! -d "$repd" ] && ( cd $(dirname $repd) && svn co file://$vard ${name}_svn )

   rm -rf $tmpd
   [ ! -d "$tmpd" ] && ( cd $(dirname $tmpd) && svn co file://$vard ${name}     )
}


adm-reset-svn(){
   local name=${1:-heprez}
   local vard=/var/scm/subversion/$name
   local tmpd=/tmp/subversion/$name
   rm -rf $tmpd
   [ ! -d "$tmpd" ] && ( cd $(dirname $tmpd) && svn co -r0 file://$vard ${name}     )
}


adm-compare-svnhg(){
   local name=${1:-env}
   local hgdir=/tmp/mercurial/$name
   local svndir=/tmp/subversion/$name

   [ ! -d "$hgdir/.hg" ] && echo hgdir $hgdir missing .hg do: adm-prep-hg $name && return

   # must start clean as svn checkout to a prior revision leaves non-empty directories hanging 
   # SVN bends over backwards by not deleting the folders
   # and as as result is a pain to deal with
   [ -d "$svndir" ] && echo $msg deleting preexisting svn working copy && rm -rf $svndir 

   local svnurl=$(adm-svnurl $name)
   local filemap=$(adm-filemap-path $name)

   local hs=($(adm-hgsvnrev $name))
   local hgrev=${hs[0]}
   local svnrev=${hs[1]}
   local opts=$(adm-opts $name)

   local cmd="compare_hg_svn.py $hgdir $svndir $svnurl --svnrev $svnrev --hgrev $hgrev --filemap $filemap $opts "
   echo $cmd
   eval $cmd

}



adm-repo(){ echo ${ADM_REPO:-env} ; }
adm-filemap-path(){   echo ~/.${1}/filemap.cfg  ; }
adm-authormap-path(){ echo ~/.${1}/authormap.cfg  ; }
adm-filemap(){
  local name=$1
  case $name in 
         env) adm-filemap-$name ;;
       g4dae) adm-filemap-$name ;;
      heprez) adm-filemap-$name ;;
     tracdev) adm-filemap-$name ;;
     opticks) adm-filemap-$name ;;
  esac
}




adm-filemap-env(){ cat << EOF
rename thho/NuWa/python/histogram/pyhist.py thho/NuWa/python/histogram/pyhist_rename_to_avoid_degeneracy.py 
EOF
}

adm-filemap-g4dae(){  cat << EOF
# g4dae exporter code history into top level of new repo
include geant4/geometry/DAE
rename geant4/geometry/DAE .
EOF
}

adm-filemap-opticks(){  cat << EOF
# split off opticks parts of env 
# guideline : minimal name changes at this stage, just filtering and moving directories around
#
# hmm if aiming for independence from env- will need lots of env machinery
# for the externals to get copied too...
#

include boost/bpo/bcfg
rename boost/bpo/bcfg bcfg

include boost/bregex
rename boost/bregex bregex

include numerics/npy 
rename numerics/npy npy

include opticks

include optix/ggeo
rename optix/ggeo ggeo

include graphics/assimpwrap
rename graphics/assimpwrap assimpwrap

include graphics/openmeshrap
rename graphics/openmeshrap openmeshrap
 
include graphics/oglrap
rename graphics/oglrap oglrap
 
include cuda/cudawrap
rename cuda/cudawrap cudawrap
 
include numerics/thrustrap
rename numerics/thrustrap thrustrap

include graphics/optixrap
rename graphics/optixrap optixrap
 
include opticksop

include graphics/ggeoview 
rename graphics/ggeoview ggeoview

include optix/cfg4 
rename optix/cfg4 cfg4

EOF
}


adm-opticks-cmake(){ cat << EOF

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(OPTICKS)

add_subdirectory(bcfg)
add_subdirectory(bregex)
add_subdirectory(npy)
add_subdirectory(opticks)
add_subdirectory(ggeo)
add_subdirectory(assimpwrap)
add_subdirectory(openmeshrap)
add_subdirectory(oglrap)
add_subdirectory(cudawrap)
add_subdirectory(thrustrap)
add_subdirectory(optixrap)
add_subdirectory(opticksop)
add_subdirectory(ggeoview)
add_subdirectory(cfg4)

EOF
}




adm-opticks(){

   cd
   rm -rf opticks

   adm-env-to-opticks

   cd opticks
   hg update

   adm-opticks-cmake > CMakeLists.txt

}



adm-filemap-g4daeview(){ cat << EOF

# see g4daeview-transmogrify 

include geant4/geometry/collada/g4daeview
rename geant4/geometry/collada/g4daeview g4daeview 

EOF
}


adm-filemap-heprez(){ cat << EOF
#placeholder
EOF
}
adm-filemap-tracdev(){ cat << EOF
#placeholder

rename annobit/trunk annobit
rename bitten/trunk bitten
rename db2trac/trunk db2trac
rename trac2latex/trunk trac2latex
rename trac2mediawiki/trunk trac2mediawiki
rename tracwiki2sphinx/trunk tracwiki2sphinx
rename xsltmacro/trunk xsltmacro
rename xsltmacro/branches/xsltmacro-blyth xsltmacro-blyth
rename xsltmacro/branches/xsltmacro-netjunki xsltmacro-netjunki

EOF
}
adm-filemap-tracdev-notes(){ cat << EON

NB when changing filemap, it is necessary 
to start from scratch by first deleting:

#. /var/scm/mercurial/tracdev 
#. /tmp/mercurial/tracdev


EON
}

adm-authormap(){ cat << EOF
blyth = Simon C Blyth <simoncblyth@gmail.com>
lint = Tao Lin <lintao51@gmail.com>
jimmy = Jimmy Ngai <jimngai@hku.hk>
maqm = Qiumei Ma <maqm@ihep.ac.cn>
thho = Taihsiang Ho <thho@hep1.phys.ntu.edu.tw>
EOF
}

adm-convert(){
   local msg="=== $FUNCNAME :"
   local name=${1:-env}
   local hgr=/var/scm/mercurial/${name:-env} 
   local svr=/var/scm/subversion/${name:-env} 
   
   #local url=http://dayabay.phys.ntu.edu.tw/repos/$repo    # NB no trunk 
   local url=file:///$svr    

   local filemap=$(adm-filemap-path $name)
   local authormap=$(adm-authormap-path $name)
   mkdir -p $(dirname $filemap)
   adm-filemap $name > $filemap
   adm-authormap $name > $authormap

   echo
   echo $msg filemap $filemap
   cat $filemap
   echo
   echo $msg authormap $authormap
   cat $authormap
   echo

   [ -d "$hgr" ] && echo $msg CAUTION destination hg repo exists already $hgr : THIS WILL BE INCREMENTAL : IF YOU CHANGED FILEMAP/OPTIONS YOU SHOULD FIRST DELETE $hgr 
   local cmd="hg convert --config convert.localtimezone=true --source-type svn --dest-type hg $url $hgr --filemap $filemap --authormap $authormap "
   echo $cmd

   local ans
   read -p "$msg enter YES to proceed " ans
   [ "$ans" != "YES" ] && return

   eval $cmd
}


adm-env-to-g4dae(){     adm-spawn g4dae env ; }
adm-env-to-opticks(){   adm-spawn opticks env ; }

adm-spawn(){

   local msg="=== $FUNCNAME :"
   local dstname=${1:-g4dae}
   local srcname=${2:-env}

   local srcdir=$HOME/$srcname
   local dstdir=$HOME/$dstname


   local dst=file://$dstdir 
   local src=file://$srcdir  

   local name=$dstname
   local filemap=$(adm-filemap-path $name)
   local authormap=$(adm-authormap-path $name)

   mkdir -p $(dirname $filemap)

   adm-filemap $name > $filemap
   adm-authormap $name > $authormap

   local cmd="hg convert --config convert.localtimezone=true --source-type hg --dest-type hg $src $dst --filemap $filemap --authormap $authormap "
   echo $cmd

   local ans
   read -p "$msg enter YES to proceed " ans
   [ "$ans" != "YES" ] && return

   eval $cmd
}



adm-verify(){

  echo  

}


