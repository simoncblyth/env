#!/bin/bash
usage(){ cat << EOU
~/env/bin/bitbucket2github.sh : migration script 
===================================================

Usage::

   cd ~/mountains # cd to the working copy repo directory  

   git remote -v  # check that repo origin is actually bitbucket
   git status     # check that the repo is clean and uptodate, notice master OR main
   git pull       # if not, get it clean and updated 

   ~/env/bin/bitbucket2github.sh    
       # run script which emits migration commands to stdout 
   
   ## before piping to shell must use github web interface 
   ## to create an empty correspondingly named repo
   ## setting it to public or private as appropriate 

   ~/env/bin/bitbucket2github.sh | sh   # if the commands look correct pipe them to shell to run them 


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

github_username=simoncblyth

repo=$(git_reponame)
test -z $repo && echo "Failed to determine git repo name." 1>&2 && exit 1

github_repo_url=git@github.com:${github_username}/${repo}.git
github_https_url=https://github.com/${github_username}/${repo}"

# curl -u "${username}:${password}" https://api.github.com/user/repos -d "{\"name\":\"$existin_repo_name\", \"private\":\"true\" , \"description\":\"Private Repo for ${username} Project - ${existin_repo_name}\" }"


echo git remote rename origin bitbucket

echo git remote add origin ${github_repo_url}
 
echo git remote set-url origin ${github_repo_url}

echo git push --set-upstream origin ${BR:-master}

#echo git remote rm bitbucket




