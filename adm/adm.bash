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


Related
--------

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


FUNCTIONS
-----------

*adm-utilities*

     Installs basic utilities: eg readline, ipython 


*adm-convert*

     Runs hg convert, migrating SVN repo to Mercurial repo 

     adm-convert env
     adm-convert heprez
     adm-convert tracdev

     Before doing this, create local SVN repo mirrors with svnsync-

*adm-svnhg name*

     Compares the SVN and converted HG repositories 
     by log comparison with timestamp matching to map between revisions.
     Makes corresponding SVN and HG checkouts for every revision, 
     compares file paths and content digests for the SVN and HG working copy.



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
    env)     echo file:///var/scm/subversion/env/trunk ;;
    heprez)  echo file:///var/scm/subversion/heprez/trunk ;;
    tracdev) echo file:///var/scm/subversion/tracdev/trunk ;;
  esac
}


adm-init-svnmirror(){
    local name=$1
    local fold=/var/scm/subversion
    mkdir -p $fold

    local repo=$fold/$name
    [ -d $repo ] &&  echo $msg repo $repo exists already && return 

    [ ! -d $repo ] &&  svnadmin create $repo
    echo '#!/bin/sh' > $repo/hooks/pre-revprop-change
    chmod +x $repo/hooks/pre-revprop-change
}



adm-hgsvnrev(){
  case $1 in 
    heprez)  echo 0 1 ;;
    tracdev) echo 0 1 ;;
    env) echo 1583 1596 ;;
    env0) echo 1600 1598 ;;
    env1) echo 3470 1598 ;;
    env2) echo 0 1 ;;
    env3) echo 724 732 ;;
  esac
}

adm-opts(){
  case $1 in 
        env) echo --skipempty --ignore-externals --clean-checkout-revs 1599,1600,1601 --known-bad-revs 1600 ;;
     heprez) echo --skipempty --ignore-externals ;;
    tracdev) echo --skipempty --ignore-externals ;;
  esac 
}

adm-svnhg(){
   local name=${1:-env}
   local hgdir=/tmp/mercurial/$name
   local svndir=/tmp/subversion/$name

   [ ! -d "$hgdir" ] && echo hgdir $hgdir missing : create with hg --cwd $(dirname $hgdir) clone /var/scm/mercurial/$name  && return

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
adm-filemap-path(){ echo ~/.${1}/filemap.cfg  ; }
adm-filemap(){
  local name=$1
  case $name in 
         env) adm-filemap-$name ;;
      heprez) adm-filemap-$name ;;
     tracdev) adm-filemap-$name ;;
  esac
}

adm-filemap-env(){ cat << EOF
rename thho/NuWa/python/histogram/pyhist.py thho/NuWa/python/histogram/pyhist_rename_to_avoid_degeneracy.py 
EOF
}
adm-filemap-heprez(){ cat << EOF
#placeholder
EOF
}
adm-filemap-tracdev(){ cat << EOF
#placeholder
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
   mkdir -p $(dirname $filemap)
   adm-filemap $name > $filemap

   echo $msg filemap $filemap
   cat $filemap

   local cmd="hg convert --config convert.localtimezone=true --source-type svn --dest-type hg $url $hgr --filemap $filemap "
   echo $cmd

   local ans
   read -p "$msg enter YES to proceed " ans
   [ "$ans" != "YES" ] && return

   eval $cmd
}




