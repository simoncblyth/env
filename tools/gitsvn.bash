# === func-gen- : tools/gitsvn fgp tools/gitsvn.bash fgn gitsvn fgh tools src base/func.bash
gitsvn-source(){   echo ${BASH_SOURCE} ; }
gitsvn-edir(){ echo $(dirname $(gitsvn-source)) ; }
gitsvn-ecd(){  cd $(gitsvn-edir); }
gitsvn-dir(){  echo $LOCAL_BASE/env/tools/gitsvn ; }
gitsvn-cd(){   cd $(gitsvn-dir); }
gitsvn-vi(){   vi $(gitsvn-source) ; }
gitsvn-env(){  elocal- ; }
gitsvn-usage(){ cat << EOU

Git SVN 
=========

* https://www.git-tower.com/blog/an-introduction-to-git-svn/

* https://gist.github.com/rickyah/7bc2de953ce42ba07116


Git Background
---------------


git merge
~~~~~~~~~~~

* https://git-scm.com/docs/git-merge

Incorporates changes from the named commits (since the time their histories
diverged from the current branch) into the current branch. This command is used
by git pull to incorporate changes from another repository and can be used by
hand to merge changes from one branch into another.


Overview
-----------

* https://git-scm.com/book/en/v2
* https://git-scm.com/docs/git-svn

...
Once tracking a Subversion repository (with any of the above methods), the Git
repository can be updated from Subversion by the fetch command and Subversion
updated from Git by the dcommit command.


Practical Tips for using git svn with large subversion repos
--------------------------------------------------------------

* http://www.janosgyerik.com/practical-tips-for-using-git-with-large-subversion-repositories/

To keep things clean, and to avoid impacting your coworkers, it might be a good
idea to keep master “pristine”. That is, never do any work on master, use it
only for interacting with the remote Subversion repository such as pull updates
and pushing local commits. Do all your work on branches, stay off the master.

::

    git checkout master
    git svn rebase  #  if you haven’t touched the master than this is like a fast-forward merge with no possibility of conflicts.


Preserving individual commits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since Subversion doesn’t have the notion of branches as Git, the easiest way to
preserve your individual commits is to rebase your branch on top of the
Subversion trunk (= master) and then push your commits to Subversion:

::

    git checkout master  # first, update from the remote trunk
    git svn rebase

    git checkout bug123  # next, rebase bug123 on top of master
    git rebase master
    git checkout master
    git merge bug123     # this should be a fast-forward

    git svn dcommit


Squashing individual commits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing special is needed here. In the rare case that the Subversion repository
has not changed since our last rebase and the bug123 branch was based on that
revision, then the merge operation will be a fast-forward by default,
preserving individual commits. Normally this is not the case, and the merged
revisions will be committed to Subversion as a single change.

::

    git checkout master  # first, update from the remote trunk
    git svn rebase

    git merge bug123     # possibly but not likely a fast-forward

    git svn dcommit


git svn show-ignore
~~~~~~~~~~~~~~~~~~~~~~~

::

    git svn show-ignore > .gitignore


Another guide : says not to merge : to keep linear history
-------------------------------------------------------------

* https://gist.github.com/rickyah/7bc2de953ce42ba07116

Do not merge your local branches, if you need to reintegrate the history of
local branches use git rebase instead.

When you perform a merge, a merge commit is created. The particular thing about
merge commits is that they have two parents, and that makes the history
non-linear. Non-linear history will confuse SVN in the case you "push" a merge
commit to the repository.

However do not worry: you won't break anything if you "push" a git merge commit to SVN.

If you do so, when the git merge commit is sent to the svn server it will
contain all the changes of all commits for that merge, so you will lose the
history of those commits, but not the changes in your code.







both svn and git remotes ?
----------------------------

* https://www.git-tower.com/help/guides/faq-and-tips/faq/svn-and-git-remotes/mac

If you need to have both a Git and a svn-remote, you should clone the svn
remote with “--prefix=svn/”.  For some reason, the git-svn team chose “origin”
to be the default prefix name; this can be problematic as you cannot have a Git
remote with name “svn” nor “origin” as they would collide. If you use the
prefix with “svn/”, you can then add your Git remote as “origin”.


git rebase
------------

* https://git-scm.com/docs/git-rebase



git svn dcommit
-----------------

Commit each diff from the current branch directly to the SVN repository, and
then rebase or reset (depending on whether or not there is a diff between SVN
and head). This will create a revision in SVN for each commit in Git.


git svn rebase
-----------------

This fetches revisions from the SVN parent of the current HEAD and rebases the
current (uncommitted to SVN) work against it.

This works similarly to svn update or git pull except that it preserves linear
history with git rebase instead of git merge for ease of dcommitting with git
svn.

This accepts all options that git svn fetch and git rebase accept. However,
--fetch-all only fetches from the current [svn-remote], and not all
[svn-remote] definitions.

Like git rebase; this requires that the working tree be clean and have no
uncommitted changes.


Getting Started with git svn
------------------------------

* https://objectpartners.com/2014/02/04/getting-started-with-git-svn/

git svn init
~~~~~~~~~~~~~~~~~

When you do git svn init, it only creates the git configuration files and does
not fetch the code right away.

workflow
~~~~~~~~~~~

At this point you have a SVN project interfaced with git. You can run git
commands on this repository such as “git commit”, “git revert”, “git diff” etc.

When you are ready to push your changes to SVN, you run the git svn dcommit
command. Before you do that, make sure you run git svn rebase.

::

    git svn rebase
    git svn dcommit 


When you run the rebase command, it will rewind your changes, fetch the latest
code from the remote server and then will apply your changes on top of it. If
you have no conflicting changes it will go smoothly. However, if there are any
conflicting changes, you will have to resolve those conflicts like in any
version control system.




Partial clones ?
---------------------

Many SVN repos have loads of branches and tags : which 
would take forever to clone

* https://stackoverflow.com/questions/14585692/how-to-use-git-svn-to-checkout-only-trunk-and-not-branches-and-tags


::

    git svn clone http://juno.ihep.ac.cn/svn/offline/trunk offline_git 

    ## this going thru all svn revs : so its real slow : aborted
    ## plus layout is : refs/remotes/git-svn 


Compare to opticks layout::

    epsilon:opticks blyth$ l .git/refs/remotes/origin/
    total 16
    -rw-r--r--  1 blyth  staff  41 Jun 17 16:23 master
    -rw-r--r--  1 blyth  staff  32 May 16 13:16 HEAD

    epsilon:opticks blyth$ cat .git/refs/remotes/origin/master
    b890d2dc7f49c1fa50fb56de5a003eaaedadc0ab
    epsilon:opticks blyth$ cat .git/refs/remotes/origin/HEAD
    ref: refs/remotes/origin/master


* https://git-scm.com/docs/git-svn




Another way ?
-----------------

* https://gist.github.com/trodrigues/1023167

::

    git svn clone -T trunk http://example.com/PROJECT






git svn init
---------------

--prefix
    Setting a prefix (with a trailing slash) is strongly encouraged in any case, as
    your SVN-tracking refs will then be located at "refs/remotes/$prefix/", which
    is compatible with Git’s own remote-tracking ref layout
    (refs/remotes/$remote/). Setting a prefix is also useful if you wish to track
    multiple projects that share a common repository. By default, the prefix is set
    to origin/.









EOU
}
gitsvn-get(){
   local dir=$(dirname $(gitsvn-dir)) &&  mkdir -p $dir && cd $dir

}
