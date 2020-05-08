# === func-gen- : tools/hg2git/hg2git fgp tools/hg2git/hg2git.bash fgn hg2git fgh tools/hg2git src base/func.bash
hg2git-source(){   echo ${BASH_SOURCE} ; }
hg2git-edir(){ echo $(dirname $(hg2git-source)) ; }
hg2git-ecd(){  cd $(hg2git-edir); }
hg2git-dir(){  echo $LOCAL_BASE/env/tools/hg2git/hg2git ; }
hg2git-cd(){   cd $(hg2git-dir); }
hg2git-vi(){   vi $(hg2git-source) ; }
hg2git-env(){  elocal- ; }
hg2git-usage(){ cat << EOU

Migrate Mecurial Repo to Git
==============================

* Older look at this in fastexport-


* https://stackoverflow.com/questions/16037787/convert-mercurial-project-to-git











 


fast export
--------------

::

    cd 
    git clone https://github.com/frej/fast-export.git
    git init git_repo
    cd git_repo
    ~/fast-export/hg-fast-export.sh -r /path/to/old/mercurial_repo
    git checkout HEAD


In case you use Mercurial < 4.6 and you got "revsymbol not found" error. 
You need to update your Mercurial or downgrade fast-export by running::

     git checkout tags/v180317 

inside ~/fast-export directory.

As an additional note, you can also pass in -A with an authors map file if you
need to map Mercurial authors to Git authors.



github importer bitbucket
--------------------------

* https://gist.github.com/mandiwise/5954bbb2e95c011885ff


github importer
-----------------

* https://help.github.com/en/articles/about-github-importer
* https://help.github.com/en/articles/importing-a-repository-with-github-importer

* https://github.blog/2019-01-07-new-year-new-github/

  Github added unlimited private repos at start of 2019


github vs gitlab vs bitbucket 
---------------------------------

* :google:`github gitlab bitbucket`

* https://usersnap.com/blog/gitlab-github/
* https://medium.com/flow-ci/github-vs-bitbucket-vs-gitlab-vs-coding-7cf2b43888a1


Features to look for
~~~~~~~~~~~~~~~~~~~~~~~

* RST support 
* Continuous Integration in free plan
* Private repos in free plan


Continuous Integration : build minutes for free
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://dev.to/mittalyashu/unlimited-free-private-repos--github-bitbucket-and-gitlab-17kj

Comparing with GitLab, I would say GitLab still wins as BitBucket only provide
50 mins/mo of build minutes whereas GitLab provides 2,000 mins/mo on shared
runners.

* :google:`gitlab free build minutes`


* https://about.gitlab.com/pricing/

2,000 CI pipeline minutes per group per month on our shared runners 


Binary Distribution : Large File Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


RST on github, gitlab
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://gist.github.com/dupuy/1855764

* https://gitlab.com/ase/ase/blob/master/README.rst
* https://gitlab.com/ase/ase/raw/master/doc/install.rst





EOU
}
hg2git-get(){
   local dir=$(dirname $(hg2git-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://repo.or.cz/fast-export.git


}
