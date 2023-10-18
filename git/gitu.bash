gitu-env(){ echo -n ; }
gitu-vi(){ vi $BASH_SOURCE ; }
gitu-usage(){ cat << EOU
gitu.bash : Tips for git usage expressed in terms of parts of workflows
=========================================================================

Other sources of git tips:

1. git- : tips expressed in a more git command centric way : so providing more detail
2. j/j (jxv) : real world git usage examples, but too long to easily find things 
   (hence this doc : aiming to glean common slices of workflow in easily to find form)

EOU
}

gitu-following-upstream-merge-tidy-and-get-back-to-main(){ cat << EOU
For example upstream is a gitlab repo where merges are controlled by web interface::

    git status            # clean : on the old development branch 
    git fetch origin      # bring local update without changing working copy 
    git checkout main     # message says how many commits behind and if can ffwd (normally can)
    git pull              # uptodate : lists files changed in those commits 
    git branch            # list branch names     
    branch=blyth-add-standalone-debug-interface-to-seven-fixture-solids
    git branch -d $branch # delete the branch that was merged
    git status            # back to clean main  

Note that a warning about deleting a branch that hasnt been merged to HEAD is fine, 
it was merged upstream.  

EOU
}


gitu-checkout-branch-created-upstream(){ cat << EOU

::

    git fetch origin
    branch=blyth-custom4-update-fixing-polarization-bug-plus-tidy
    git checkout -b $branch origin/$branch


EOU
}







