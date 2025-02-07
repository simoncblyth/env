#!/bin/bash
usage(){ cat << EOU
~/env/bin/bitbucket2github.sh : migration script 
===================================================

See: ~/home/notes/bitbucket/migrate_repos_from_bitbucket_to_github.rst

Migration SOP
--------------

1. clone repo from bitbucket to laptop
2. check the email addresses associated to the commits::

   git log --pretty=format:%ae | sort -u

3. if any email addresses not yet added to github, do so. Currently have::

    simon.c.blyth@gmail.com
    simoncblyth@gmail.com    
    blyth@hep1.phys.ntu.edu.tw
    blyth@ihep.ac.cn

4. use github web interface to create repo with same name as the bitbucket one being migrated

5. take a look for issues/complications, notice if master OR main::

    cd ~/sphinxtest
    git remote -v  # check that repo origin is actually bitbucket
    git status     # check that the repo is clean and uptodate, notice master OR main
    git pull       # if not clean, make it clean and updated 

6. run this migration script from the repo directory::

    ~/env/bin/bitbucket2github.sh       ## list commands
    ~/env/bin/bitbucket2github.sh | sh  ## run them 

    BR=main ~/env/bin/bitbucket2github.sh       ## list commands
    BR=main ~/env/bin/bitbucket2github.sh | sh  ## run them 

7. check github web interface for the migrated repo


Details
--------

master OR main
   older repos use default branch name of master, 
   annoyingly newer repos use main
   have to adapt for main with::

      BR=main ~/env/bin/bitbucket2github.sh

 -u (or --set-upstream) 
     flag sets the remote origin as the upstream reference. 
     This allows you to later perform git push and git pull commands 
     without having to specify an origin since we always want GitHub in this case.


EOU
}

git_reponame(){  git config --local remote.origin.url|sed -n 's#.*/\([^.]*\)\.git#\1#p' ; }

repo=$(git_reponame)
test -z $repo && echo "Failed to determine git repo name." 1>&2 && exit 1

github_username=simoncblyth
github_ssh_url=git@github.com:${github_username}/${repo}.git
github_https_url=https://github.com/${github_username}/${repo}

cat << EOC

git remote -v

git remote rename origin bitbucket

git remote add origin ${github_ssh_url}
 
git remote set-url origin ${github_ssh_url}

git remote -v

git push --set-upstream origin ${BR:-master}

open ${github_https_url}

EOC


