Welcome to env
=================

.. contents:: :local:


Overview
---------

The *env* provides mostly bash and python functions for many tasks related to getting,
configuring, installing, running, logging and validating software.
Initially setup for the Daya Bay collaboration however much of the functionality in
general in nature and could be useful to anyone.

In a former life the *env* was a Trac/SVN repository, it has now
been migrated to BitBucket/Mercurial.

Get/update your env with the below, this will create/update the folder $HOME/env::

    cd $HOME ; 
    hg clone ssh://hg@bitbucket.org/simoncblyth/env   # using SSH to allow passwordless access via keys
    hg pull

Hook up your bash shell with the env by adding the below to your .bash_profile::

    export ENV_HOME=$HOME/env      
    env-(){  . $ENV_HOME/env.bash  && env-env $* ; }
    env-    
    
The *env-* causes the definition of many bash precursor functions.


Making changes to env from Bitbucket
--------------------------------------

In order for you to make changes to env on bitbucket.

Once only setup
~~~~~~~~~~~~~~

#. create a free bitbucket account (you will need to provide an email address)
   https://bitbucket.org

#. fork https://bitbucket.org/simoncblyth/env into your account 
   using the bitbucket web interface

#. use mercurial to clone your fork onto machines using env

passwordless access using ssh keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collect together the public keys on all machines that need to 
clone the repository and use bitbucket web interface to add the keys
to your bitbucket account.::

    (adm_env)delta:~ blyth$ scp C2:.ssh/id_rsa.pub C2.id_rsa.pub
    (adm_env)delta:~ blyth$ cat C2.id_rsa.pub | pbcopy
    (adm_env)delta:~ blyth$ 
    (adm_env)delta:~ blyth$ scp N:.ssh/id_rsa.pub N.id_rsa.pub
    (adm_env)delta:~ blyth$ cat N.id_rsa.pub | pbcopy
    (adm_env)delta:~ blyth$ 
    (adm_env)delta:~ blyth$ scp G:.ssh/id_rsa.pub G.id_rsa.pub
    (adm_env)delta:~ blyth$ cat G.id_rsa.pub | pbcopy



For each change
~~~~~~~~~~~~~~~~

#. commit to local repository with Mercurial and push to bitbucket fork up in the cloud

   ::

       hg commit -m "informative but brief message"
       hg push 

#. to share the change (best to do this to avoid divergence)
   make pull requests to me using bitbucket web interface


Getting current: hg pull then hg update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For passwordless, need to run the ssh agent (see ssh--agent-start)::

    [blyth@cms01 env]$ hg pull
    Enter passphrase for key '/home/blyth/.ssh/id_rsa': 
    pulling from ssh://hg@bitbucket.org/simoncblyth/env
    searching for changes
    adding changesets
    adding manifests
    adding file changes
    added 2 changesets with 8 changes to 7 files
    (run 'hg update' to get a working copy)

    [blyth@cms01 env]$ hg up
    7 files updated, 0 files merged, 0 files removed, 0 files unresolved



Possible future simplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bitbucket does actually have "teams" which allow 
team members to all push into a single repository.
I'll investigate this possibility further.

* http://blog.bitbucket.org/2012/05/30/bitbucket-teams/
* https://confluence.atlassian.com/display/BITBUCKET/Bitbucket+Teams




Hierarchical Organization of functions
---------------------------------------

Functions ending in hyphens such as *swig-* and *python-* are precursor functions
that on running lead to the definition of several other functions within these
namespaces and the running of the corresponding *-env* function. In this way the
functions are insured of a particular environment while minimizing namespace
pollution.

Dependencies between sets of functions are setup by using the precursors where
they are needed, using a kitchensink approach is deprecated as it is then
unclear of what depends on what making errors harder to trace and making
modifications more difficult.

The top level "precursors" are defined in *env.bash* and "sub-precursors" should
be defined in *.bash* named after the top level folder like *swig/swig.bash* or
*python/python.bash* etc...

After running the precursors you can use tab completion in the shell to see the
functions that have been defined::

      swig-<tab>

Thus a sequence of commands like::

     swig-
     swigbuild-
     swigbuild-usage
     swigbuild-again

gets you from an environment with only top level precursor functions to one
with the specific functions you need and no more.


