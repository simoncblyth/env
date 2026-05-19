#!/usr/bin/env bash
usage(){ cat << EOU
~/env/simoncblyth.github.io/make.sh
====================================

Context:

* ~/e "env" repo contains sources (eg .txt for all presentations and index.txt)
  that yield html by running scripts from env repo.

* derived html is managed in ~/s "simoncblyth" repo together with images
  referenced from those presentations

* ~/s "simoncblyth" repo is pushed to github and bitbucket for ".io" web presentation

* on both zeta laptop and Workstation A, the ~/s repo dir is symbolically linked::

    s -> /usr/local/simoncblyth.github.io


Workflow to update index pages on github and bitbucket
----------------------------------------------------------

Index pages:

* https://simoncblyth.github.io
* https://simoncblyth.bitbucket.io


Formerly did this with Makefile on macOS laptop.
Now moved html generation of index to Linux workstation A.


1. edit sources in env repo::

   cd ~/env/simoncblyth.github.io
   vi index.txt

2. convert the RST .txt to html with rst2html by running the make.sh script (needs miniconda env with docutils, eg use "lco" bash function)::

   ~/env/simoncblyth.github.io/make.sh

3. brief check of the html on Linux browser, note formatting not so good - no problem

4. From A, push the html to servers::

    s   # cd ~/s ## which is symbolic link to /usr/local/simoncblyth.github.io

    git add index.html
    git commit -m "update index with presentation... "
    git remote -v

    git pull              ## make sure uptodate before push
    git push              ## default origin       https://simoncblyth.github.io
    git push bitbucket    ## other                https://simoncblyth.bitbucket.io
    ./rsync_put_to_W.sh   ## juno.ihep.ac.cn      https://juno.ihep.ac.cn/~blyth/


5. check resulting web pages have updated, github and bitbucket can take ~10 mins before change appears::

   gio open https://simoncblyth.github.io
   gio open https://simoncblyth.bitbucket.io
   gio open https://juno.ihep.ac.cn/~blyth/

   ## browsers cache aggressively
   ## simpler to check an update with a different browser than to try to defeat the cache



If remotes not already configured
-----------------------------------

::

    git remote -v
    git remote add bitbucket git@bitbucket.org:simoncblyth/simoncblyth.bitbucket.io.git
    git remote -v


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

case $(uname) in
  Darwin) RST2HTML=rst2html-3.13 ;;
  Linux)  RST2HTML=rst2html      ;;
esac

if [ "$(uname)" == "Linux" ]; then
   open() {
        :  make.sh
        : miniconda does not play nice with gio open - have to override gui path
        XKB_CONFIG_ROOT=/usr/share/X11/xkb gio open "$@"
    }
fi


GITHUB_HTDOCS=/usr/local/simoncblyth.github.io


defarg="info_check_copy_rst2html_ls_open"
arg=${1:-$defarg}

vv="BASH_SOURCE RST2HTML GITHUB_HTDOCS OPEN OPEN_NOTE"

if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ check ]]; then
    if [ ! -d "$GITHUB_HTDOCS" ]; then
       echo $BASH_SOURCE - ERROR GITHUB_HTDOCS $GITHUB_HTDOCS
       exit 1
    fi
    if command -v $RST2HTML &> /dev/null; then
        echo "$BASH_SOURCE - INFO - RST2HTML $RST2HTML is installed and available in PATH"
    else
        echo "$BASH_SOURCE - FATAL - RST2HTML $RST2HTML is NOT installed - try \"lco\" bash funtion to activate miniconda python env with docutils " >&2
        exit 1
    fi
fi

if [[ "$arg" =~ copy ]]; then
    cp custom.css $GITHUB_HTDOCS/custom.css
    [ $? -ne 0 ] && echo $BASH_SOURCE -  copy ERROR && exit 1
fi

if [[ "$arg" =~ rst2html ]]; then
    $RST2HTML --stylesheet=$GITHUB_HTDOCS/custom.css index.txt $GITHUB_HTDOCS/index.html
    [ $? -ne 0 ] && echo $BASH_SOURCE -  rst2html ERROR && exit 1
fi

if [[ "$arg" =~ ls ]]; then
   echo ls -alst $GITHUB_HTDOCS
   ls -alst $GITHUB_HTDOCS
   [ $? -ne 0 ] && echo $BASH_SOURCE - ls ERROR && exit 1
fi

if [[ "$arg" =~ open ]]; then
   open $GITHUB_HTDOCS/index.html
   [ $? -ne 0 ] && echo $BASH_SOURCE -  open ERROR && exit 1
fi

