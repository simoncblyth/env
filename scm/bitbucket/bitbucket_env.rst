Bitbucket Env
==============


Env Bitbucket Usage
-----------------------------

In order for you to make changes to env on bitbucket.

Once only setup
~~~~~~~~~~~~~~

#. create a free bitbucket account (you will need to provide an email address)
   https://bitbucket.org

#. fork https://bitbucket.org/simoncblyth/env into your account 
   using the bitbucket web interface
  (env is currently private as I keep deleting it while 
   still testing the svn to hg conversion)

#. use mercurial to clone your fork onto machines using env


For each change
~~~~~~~~~~~~~~~~

#. commit to local repository with Mercurial and push to bitbucket fork up in the cloud

   ::

       hg commit -m “…”
       hg push 

#. to share the change (best to do this to avoid divergence)
   make pull requests to me using bitbucket web interface


Possible future simplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bitbucket does actually have “teams” which allow 
team members to all push into a single repository.
I'll investigate this possibility further.

* http://blog.bitbucket.org/2012/05/30/bitbucket-teams/
* https://confluence.atlassian.com/display/BITBUCKET/Bitbucket+Teams





