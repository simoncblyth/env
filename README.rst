Welcome to env
=================

The *env* provides mostly bash and python functions for many tasks related to getting,
configuring, installing, running, logging and validating software.
Initially setup for the Daya Bay collaboration however much of the functionality in
general in nature and could be useful to anyone.

In a former life the *env* was a Trac/SVN repository, it has now
been migrated to BitBucket/Mercurial.

Get/update your env with the below, this will create/update the folder $HOME/env::

    cd $HOME ; 
    hg clone ssh://hg@bitbucket.org/simoncblyth/env
    hg pull

Hook up your bash shell with the env by adding the below to your .bash_profile::

    export ENV_HOME=$HOME/env      
    env-(){  . $ENV_HOME/env.bash  && env-env $* ; }
    env-    
    The env- causes the definition of many bash functions.


Hierarchical Organization of functions
---------------------------------------

Functions ending in hyphen `-` such as `swig-` and `python-` are precursor functions
that on running lead to the definition of several other functions within these
namespaces and the running of the corresponding `*-env` function. In this way the
functions are insured of a particular environment while minimizing namespace
pollution.

Dependencies between sets of functions are setup by using the precursors where
they are needed, using a kitchensink approach is deprecated as it is then
unclear of what depends on what making errors harder to trace and making
modifications more difficult.

The top level "precursors" are defined in `env.bash` and "sub-precursors" should
be defined in `.bash` named after the top level folder like `swig/swig.bash` or
`python/python.bash` etc...

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


