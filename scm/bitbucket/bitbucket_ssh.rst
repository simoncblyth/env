Bitbucket SSH 
===================

https/http clones from bitbucket are hanging 
----------------------------------------------

was not patient enough
~~~~~~~~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:src blyth$ hg clone https://scb-@bitbucket.org/scb-/chroma
    destination directory: chroma
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    ^Ctransaction abort!
    rollback completed
    interrupted!


after several minutes it succeeds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:src blyth$ hg clone https://scb-@bitbucket.org/scb-/chroma
    destination directory: chroma
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 233 changesets with 546 changes to 165 files (+2 heads)
    updating to branch default
    159 files updated, 0 files merged, 0 files removed, 0 files unresolved
    (chroma_env)delta:src blyth$ 


Try setting up SSH
--------------------------

::

    (chroma_env)delta:bitbucket blyth$ ssh -T git@bitbucket.org
    The authenticity of host 'bitbucket.org (131.103.20.168)' can't be established.
    RSA key fingerprint is 97:8c:1b:f2:6f:14:6b:5c:3b:ec:aa:46:46:74:7c:40.
    Are you sure you want to continue connecting (yes/no)? yes
    Warning: Permanently added 'bitbucket.org,131.103.20.168' (RSA) to the list of known hosts.
    Permission denied (publickey).

    (chroma_env)delta:bitbucket blyth$ ssh -T hg@bitbucket.org
    Warning: Permanently added the RSA host key for IP address '131.103.20.167' to the list of known hosts.
    Permission denied (publickey).


follow along SSH setup
-------------------------

* https://confluence.atlassian.com/pages/viewpage.action?pageId=270827678

I already use SSH, so can skip steps 1-4

Step 5. Enable SSH compression for Mercurial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `~/.hgrc`::

   [ui]
   ssh = ssh -C


Step 6. Install the public key on your Bitbucket account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paste public key into bitbucket webinterface and label as `Delta RSA Public Key`::

    pbcopy < ~/.ssh/id_rsa.pub


try a clone via ssh
~~~~~~~~~~~~~~~~~~~~~

Huh, its no faster than https::

    (chroma_env)delta:src blyth$ mv chroma chroma_https
    (chroma_env)delta:src blyth$ hg clone ssh://hg@bitbucket.org/scb-/chroma
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa':                      ## agent not running ?
    destination directory: chroma
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 233 changesets with 546 changes to 165 files (+2 heads)
    updating to branch default
    159 files updated, 0 files merged, 0 files removed, 0 files unresolved
    (chroma_env)delta:src blyth$ 


Compare clones
~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:src blyth$ diff -r --brief chroma_https chroma
    Files chroma_https/.hg/dirstate and chroma/.hg/dirstate differ
    Files chroma_https/.hg/hgrc and chroma/.hg/hgrc differ
    Files chroma_https/.hg/undo.desc and chroma/.hg/undo.desc differ
    (chroma_env)delta:src blyth$ 

    (chroma_env)delta:src blyth$ diff -r  chroma_https chroma
    Binary files chroma_https/.hg/dirstate and chroma/.hg/dirstate differ
    diff -r chroma_https/.hg/hgrc chroma/.hg/hgrc
    2c2
    < default = https://scb-@bitbucket.org/scb-/chroma
    ---
    > default = ssh://hg@bitbucket.org/scb-/chroma
    diff -r chroma_https/.hg/undo.desc chroma/.hg/undo.desc
    3c3
    < https://scb-@bitbucket.org/scb-/chroma
    ---
    > ssh://hg@bitbucket.org/scb-/chroma
    (chroma_env)delta:src blyth$ 



Step 7. Change your repo from HTTPS to the SSH protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice different username placement in the URLs::

    https://scb-@bitbucket.org/scb-/chroma
    ssh://hg@bitbucket.org/scb-/chroma


No need to change `chroma/.hg/hgrc` as already setup for SSH.



Check SSH setup
~~~~~~~~~~~~~~~~

* `ssh -T` Disables pseudo-tty allocation


::

    (chroma_env)delta:chroma blyth$ ssh -T hg@bitbucket.org
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    logged in as scb-.

    You can use git or hg to connect to Bitbucket. Shell access is disabled.
    (chroma_env)delta:chroma blyth$ 


::

    (chroma_env)delta:chroma blyth$ ssh--agent-start
    ...

    (chroma_env)delta:chroma blyth$ ssh -T hg@bitbucket.org
    logged in as scb-.

    You can use git or hg to connect to Bitbucket. Shell access is disabled.
    (chroma_env)delta:chroma blyth$ 






