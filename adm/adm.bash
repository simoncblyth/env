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


env access
~~~~~~~~~~~

Also *adm-env-ln*


FUNCTIONS
-----------

*adm-utilities*
     Installs basic utilities: eg readline, ipython 


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



adm-env-svnhg(){

   local repo=env
   local hgdir=/tmp/mercurial/$repo
   local svndir=/var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637
   local svnrev=1600
   local hgrev=1598

   local degenerates=~/.$repo/degenerates.cfg
   local filemap=$(adm-filemap)

   [ ! -d ~/.$repo ] && mkdir ~/.$repo 
   [ ! -f $degenerates ] && adm-env-degenerates > $degenerates

   compare_hg_svn.py $hgdir $svndir --svnrev $svnrev --hgrev $hgrev -A  --skipempty --degenerates $degenerates 

}

adm-repo(){ echo ${ADM_REPO:-env} ; }
adm-filemap-path(){ echo ~/.$(adm-repo)/filemap.cfg  ; }
adm-filemap(){ 
  local repo=$(adm-repo)
  case $repo in 
     env) adm-filemap-$repo ;;
  esac
}
adm-filemap-env(){ cat << EOF
rename trunk/thho/NuWa/python/histogram/pyhist.py trunk/thho/NuWa/python/histogram/pyhist_avoiding_case_degeneracy.py
EOF
}


adm-env-convert(){
   local repo=${1:-env}
   local hgr=/var/scm/mercurial/${repo:-env} 
   #local url=http://dayabay.phys.ntu.edu.tw/repos/$repo/trunk
   local url=http://dayabay.phys.ntu.edu.tw/repos/$repo

   local filemap=$(adm-filemap-path)
   [ ! -f "$filemap" ] && adm-filemap > $filemap
   local cmd="hg convert --config convert.localtimezone=true --source-type svn --dest-type hg $url $hgr --filemap $filemap"
   echo $cmd
   eval $cmd
}

adm-env-degenerates(){ cat << EOF
/thho/NuWa/python/histogram/pyhist.py
/thho/NuWa/python/histogram/PyHist.py
EOF
}




