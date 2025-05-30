adm-src(){      echo adm/adm.bash ; }
adm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(adm-src)} ; }
adm-vi(){       vi $(adm-source) ; }
adm-usage(){ cat << \EOU

ADM : Repository SysAdmin with Python Virtualenv 
=================================================

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


adm-spawn : Spawns subset of repo into a new one, with history
-----------------------------------------------------------------

See example : adm-opticks

Preparing *env* for spawning *opticks* 
-----------------------------------------

* move opticks projs up to toplevel, changing precursors
* move bash functions for opticks externals into new top level externals folder

* Migrate infrastructure such as proj precursors from env.bash into opticks.bash  

svnsync
-----------

* http://www.cardinalpath.com/how-to-use-svnsync-to-create-a-mirror-backup-of-your-subversion-repository/

... create a mirror repository on another server, and use the svnsync program to 
create a replica of your primary Subversion repository,

* http://svnbook.red-bean.com/en/1.7/svn.ref.svnsync.html



What mappings to make in the conversion
-----------------------------------------

As few as possible. 

Easier to do renames in env (in an audited fashion) 
with standard "hg mv" prior to spawn.

Transitionally the env sources will continue to exist beyond the spawn 
for easier mapping from old to new its better to do the repositioning
within env prior to the spawn.

Avoid complicated filemap, it should just be
inclusion of top level files and the project folders. 

   include opticks.bash
   include CMakeLists.txt
   include Makefile

   include externals
   include sysrap
   include boostrap
   include npy
   include optickscore 
   include ggeo
   ...   


How to verify a spawed repo ?
-------------------------------

1. full opticks gathering externals and doing clean build operational 
   out of the spawned repo 

How to verify history ?
----------------------------

Local web interface to browse history::

    cd ~/opticks
    hg serve

Meanwhile::

    open http://delta.local:8000

Issues ~/opticks history
--------------------------

Non-relevant tracts:

* from prior use of "externals" and from the sphinxbuild Makefile

Missing:

* cmake folders
* history from the pkgs moved to top


opticks push to bitbucket  
---------------------------

::

    delta:opticks blyth$ 
    delta:opticks blyth$ hg push ssh://hg@bitbucket.org/simoncblyth/opticks
    pushing to ssh://hg@bitbucket.org/simoncblyth/opticks
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 1153 changesets with 14073 changes to 3080 files
    delta:opticks blyth$ 


*hg convert* config 
-----------------------

::

    delta:~ blyth$ hg help convert
    hg convert [OPTION]... SOURCE [DEST [REVMAP]]
    ...
    The Mercurial source recognizes the following configuration options, which
    you can set on the command line with "--config":

    convert.hg.ignoreerrors
                  ignore integrity errors when reading. Use it to fix
                  Mercurial repositories with missing revlogs, by converting
                  from and to Mercurial. Default is False.
    convert.hg.saverev
                  store original revision ID in changeset (forces target IDs
                  to change). It takes a boolean argument and defaults to
                  False.
    convert.hg.revs
                  revset specifying the source revisions to convert.


Check hgext.convert source
-----------------------------

::

    In [4]: import hgext.convert as _

    In [6]: _??

    In [8]: _.__file__
    Out[8]: '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/hgext/convert/__init__.pyc'


/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/hgext/convert/hg.py::

    253         # Restrict converted revisions to startrev descendants
    254         startnode = ui.config('convert', 'hg.startrev')
    255         hgrevs = ui.config('convert', 'hg.revs')
    256         if hgrevs is None:
    257             if startnode is not None:
    258                 try:
    259                     startnode = self.repo.lookup(startnode)
    260                 except error.RepoError:
    261                     raise util.Abort(_('%s is not a valid start revision')
    262                                      % startnode)
    263                 startrev = self.repo.changelog.rev(startnode)
    264                 children = {startnode: 1}
    265                 for r in self.repo.changelog.descendants([startrev]):
    266                     children[self.repo.changelog.node(r)] = 1
    267                 self.keep = children.__contains__
    268             else:
    269                 self.keep = util.always
    270             if rev:
    271                 self._heads = [self.repo[rev].node()]
    272             else:
    273                 self._heads = self.repo.heads()
    274         else:
    275             if rev or startnode is not None:
    276                 raise util.Abort(_('hg.revs cannot be combined with '
    277                                    'hg.startrev or --rev'))
    278             nodes = set()
    279             parents = set()
    280             for r in scmutil.revrange(self.repo, [hgrevs]):
    281                 ctx = self.repo[r]
    282                 nodes.add(ctx.node())
    283                 parents.update(p.node() for p in ctx.parents())
    284             self.keep = nodes.__contains__
    285             self._heads = nodes - parents



Convert Trawling
------------------

* http://stackoverflow.com/questions/3643313/mercurial-copying-one-file-and-its-history-to-another-repository


*hg log --follow* needed to follow thru renames
--------------------------------------------------

::

    delta:opticksnpy blyth$ hg shortlog NPY.hpp
    6611e08d62cc | 2016-07-02 15:32:32 +0800 | simoncblyth: move numerics/npy up to top level opticksnpy


    delta:opticksnpy blyth$ hg shortlog -f NPY.hpp
    6611e08d62cc | 2016-07-02 15:32:32 +0800 | simoncblyth: move numerics/npy up to top level opticksnpy
    aacb7eba15ae | 2016-06-24 12:57:25 +0800 | simoncblyth: avoid MSVC template complications in oglrap- Renderer  ...
    17fef7662265 | 2016-06-17 21:08:45 +0800 | simoncblyth: testing usage of NPY subset in NPYClient npc-
    ...
    2f825c82a3a8 | 2015-04-15 13:59:36 +0800 | simoncblyth: G4StepNPY a friend class of NPY to avoid inheritance hassles, dumping CerenkovStep NPY
    482d9f68f6f6 | 2015-04-15 12:16:49 +0800 | simoncblyth: NPY handling in the numpydelegate with NumpyEvt, C++ equivalent of env/g4dae/types.py


::

    delta:opticksnpy blyth$ hg flog -f NPY.hpp

    482d9f68f6f6 | 2015-04-15 12:16:49 +0800 | simoncblyth: NPY handling in the numpydelegate with NumpyEvt, C++ equivalent of env/g4dae/types.py
      boost/basio/numpyserver/CMakeLists.txt
      boost/basio/numpyserver/NPY.hpp
      boost/basio/numpyserver/NumpyEvt.cpp
      boost/basio/numpyserver/NumpyEvt.hpp
      boost/basio/numpyserver/main.cpp
      boost/basio/numpyserver/numpydelegate.cpp
      boost/basio/numpyserver/numpydelegate.hpp
      boost/basio/numpyserver/numpyserver.bash
      boost/basio/numpyserver/tests/CMakeLists.txt
      boost/basio/numpyserver/tests/NPYTest.cc
      boost/basio/numpyserver/tests/NumpyServerTest.cc
      graphics/ggeoview/main.cc


hg revsets
-----------
 
* https://www.selenic.com/blog/?p=744


What to include in spawned opticks repo 
-----------------------------------------

* Everything needed to build Opticks, including bash functions for externals.
* bash function infrastructure

What to exclude 
------------------

The point of spawned Opticks repo is **CLARITY** and **SIMPLICITY** :
so exclude as much as possible.
*env* and *opticks* repos will coexist so no pressure to include.

* dev notes not relevant to users
* experimental stuff, eg Windows psm1 modules 

* Sphinx documentation sources and Makefile

  * the Makefile brings with it a long irrelevant history  


Background on *hg convert*
----------------------------

* :google:`hg convert`


wiki/ConvertExtension
~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/ConvertExtension

  This extension is distributed with Mercurial. 

Note: 

When converting to Mercurial, the destination working directory is used as a
temporary storage for file revisions but is not updated. hg status lists all
these temporary files as unknown. Purge them and update to get a correct view
of the converted repository.

::

    delta:env blyth$ find . -type f -depth 1
    ./.hgignore
    ./__init__.py
    ./__init__.pyc
    ./CMakeLists.txt
    ./conf.py
    ./env.bash
    ./index.rst
    ./install.rst
    ./main.scons
    ./Makefile
    ./opticks-failed-build.bash
    ./opticks.bash
    ./opticksdata.bash
    ./opticksdev.bash
    ./optickswin.bash
    ./README.rst
    ./sweep.py
    ./TODO.rst


wiki/ConvertExtensionImplementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/ConvertExtensionImplementation


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
adm-sdir(){ echo $(env-home)/adm ; }
adm-dir(){ echo $(local-base)/env/adm_env ; }
adm-sitedir(){ echo $(adm-dir)/lib/python2.7/site-packages ; }
adm-sitedir-cd(){ cd $(adm-sitedir) ; }

adm-scd(){  cd $(adm-sdir); }
adm-cd(){  cd $(adm-dir); }

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

adm-svnreposurl(){ echo ${ADM_SVNREPOSURL:-http://dayabay.phys.ntu.edu.tw.notused.always.envvar/repos} ; }

adm-svnmirror-init-notes(){ cat << EON

(adm_env)delta:~ blyth$ svnsync help init
initialize (init): usage: svnsync initialize DEST_URL SOURCE_URL

Initialize a destination repository for synchronization from
another repository.

If the source URL is not the root of a repository, only the
specified part of the repository will be synchronized.

The destination URL must point to the root of a repository which
has been configured to allow revision property changes.  In
the general case, the destination repository must contain no
committed revisions.  Use --allow-non-empty to override this
restriction, which will cause svnsync to assume that any revisions
already present in the destination repository perfectly mirror
their counterparts in the source repository.  (This is useful
when initializing a copy of a repository as a mirror of that same
repository, for example.)

You should not commit to, or make revision property changes in,
the destination repository by any method other than 'svnsync'.
In other words, the destination repository should be a read-only
mirror of the source repository.

EON
}

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
   #local repd=$HOME/${name}_svn
   local tmpd=/tmp/subversion/$name

   [ ! -d "$vard" ] && echo $msg ERROR no $vard && return
   #[ ! -d "$repd" ] && ( cd $(dirname $repd) && svn co file://$vard ${name}_svn )

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
  shift 
  case $name in 
         env) adm-filemap-$name ;;
       g4dae) adm-filemap-$name ;;
      heprez) adm-filemap-$name ;;
     tracdev) adm-filemap-$name ;;
     opticks) adm-filemap-$name  $* ;;
    workflow) adm-filemap-$name  $* ;;
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

adm-filemap-opticks(){  $ENV_HOME/adm/opticks_filemap.py $* ; }

adm-filemap-workflow(){ cat << EOF
# placeholder $FUNCNAME
EOF
}

adm-opticks(){

   type $FUNCNAME

   local ans
   read -p "$msg Are you sure ? enter YES to proceed " ans
   [ "$ans" != "YES" ] && return

   cd
   rm -rf opticks

   local firstrev=4910

   adm-spawn opticks env $firstrev 

   if [ -d "opticks" ]; then 
      cd opticks
      hg update
   else
      echo $msg FAILED TO SPAWN
   fi
}


adm-spawn-notes(){ cat << EON

adm-spawn vs adm-convert
===========================

adm-convert 
    full history conversion between 
    standardly positioned svn and hg repos:

    * /var/scm/subversion/name 
    * /var/scm/mercurial/name

adm-spawn 

    spawns between two hg repos in HOME, 
    with firstrev argument to cut history 
 
    As demonstrated by ~/env/adm/opticks_filemap.py
    it is possible to exercise great control over 
    what gets spawned.


For ease of validation of the conversion 
it seems a good idea to keep the convert svn->hg 
as simple as possible, doing any complicated splitting 
in a subsequent hg->hg spawn step.

EON
}

adm-env-to-g4dae(){ adm-spawn g4dae env ; }
adm-spawn(){

   local msg="=== $FUNCNAME :"
   local dstname=${1:-destination}
   local srcname=${2:-env}
   local firstrev=${3:-0}

   local dstdir=$HOME/$dstname
   local srcdir=$HOME/$srcname

   local dst=file://$dstdir 
   local src=file://$srcdir  

   echo $msg src $src dst $dst firstrev $firstrev  

   local filemap=$(adm-filemap-path $dstname)
   local authormap=$(adm-authormap-path $dstname)

   mkdir -p $(dirname $filemap)

   adm-filemap   $dstname --firstrev $firstrev > $filemap
   adm-authormap $dstname > $authormap

   local cmd="hg convert --config convert.hg.startrev=$firstrev --config convert.localtimezone=true --source-type hg --dest-type hg $src $dst --filemap $filemap --authormap $authormap "
   echo $cmd

   local ans
   read -p "$msg enter YES to proceed " ans
   [ "$ans" != "YES" ] && return

   eval $cmd
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

adm-authormap-env(){ cat << EOF
blyth = Simon C Blyth <simoncblyth@gmail.com>
lint = Tao Lin <lintao51@gmail.com>
jimmy = Jimmy Ngai <jimngai@hku.hk>
maqm = Qiumei Ma <maqm@ihep.ac.cn>
thho = Taihsiang Ho <thho@hep1.phys.ntu.edu.tw>
EOF
}

adm-authormap-scb(){ cat << EOF
blyth = Simon C Blyth <simoncblyth@gmail.com>
EOF
}


adm-authormap(){ 
  local name=$1
  shift 
  case $name in 
         env) adm-authormap-$name ;;
           *) adm-authormap-scb ;;
  esac
}

adm-convert(){
   local msg="=== $FUNCNAME :"
   local name=${1:-env}
   local hgr=/var/scm/mercurial/${name:-env} 
   local svr=/var/scm/subversion/${name:-env} 
   
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
   [ "$ans" != "YES" ] && echo $msg SKIPPING... && return

   echo $msg proceeding
   eval $cmd
}



adm-verify(){
  echo  
}


