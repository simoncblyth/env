Welcome to env
=================


Hub Page and env notes
---------------------------

* http://simoncblyth.bitbucket.org
* http://simoncblyth.bitbucket.org/env/notes/

Opticks Has Moved
--------------------

All Opticks developments should now happen over in

* http://bitbucket.org/simoncblyth/opticks


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


Mercurial + Bitbucket Setup 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Mercurial preliminaries, create a `~/.hgrc` containing the username and email
   you will use to register with bitbucket, eg::

        [ui]
        username = Simon Blyth <simoncblyth@gmail.com>
        ssh = ssh -C

#. create a free bitbucket account (you will need to provide the same email address as in .hgrc)
   https://bitbucket.org


Gather SSH public keys and paste into Bitbucket web interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collect together the public keys of all machines that need to 
clone the repository and use bitbucket web interface to add the keys
to your bitbucket account.::

    scp C2:.ssh/id_rsa.pub C2.id_rsa.pub
    cat C2.id_rsa.pub | pbcopy
    # paste into browser form
    
    scp N:.ssh/id_rsa.pub N.id_rsa.pub
    cat N.id_rsa.pub | pbcopy
    # paste into browser form


Setup steps for envsys team members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
Team members of envsys have direct push privileges into 
the env repository http://bitbucket.org/simoncblyth/env

#. check your bitbucket account is one of the envsys team members

   * https://bitbucket.org/envsys/profile/members 
 
#. clone the repository into your directory using ssh urls:: 

   hg clone ssh://hg@bitbucket.org/simoncblyth/env
 

Setup steps for non-envsys team members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non team members will need to

#. fork the repository into their bitbucket account using 
   the webinterface

#. clone the forked repository into their directory::

   hg clone ssh://hg@bitbucket.org/yourusername/env


For each change
~~~~~~~~~~~~~~~~

#. commit to local repository with Mercurial and push to bitbucket original or fork 
   up in the cloud

   ::

       hg commit -m "informative but brief message"
       hg paths      # check the default path where pushes will go 
       hg push 


Extra steps for non team members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. make pull requests to the original env 
   repository using bitbucket webinterface

#. wait for original env administrator to perform the pull



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

Testing envsys team access to env 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Juggling two bitbucket identities, is not generally recommended, but can do it for
testing team access. Create an `~/.ssh/config` section for bitbucket::

    # for correct bitbucket identification of commit need to set ~/.hgrc ui/username to simoncblyth@
    host BB
         user hg
         hostname bitbucket.org
         Compression yes
         IdentityFile /Users/blyth/.ssh/id_rsa

    # for correct bitbucket identification of commit need to set ~/.hgrc ui/username to simon.cblyth@
    host BBTEAM
         user hg
         hostname bitbucket.org
         Compression yes
         IdentityFile /Users/blyth/.ssh/id_dsa


Then can clone with the below. Thus is advantageous when switching between identities as
can control the SSH key that is used::

     hg clone ssh://BB/simoncblyth/env



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


