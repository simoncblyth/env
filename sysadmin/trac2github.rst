Trac2Github 
=============

Migration of code and bug tracker issues stored in SVN+Trac to Github 

SVN to git
---------------

 * want to make a one time transition, with standard git after the change 
 * retain history in migration, expect there are pre-cooked solutions for this
 * http://stackoverflow.com/questions/79165/how-to-migrate-svn-with-history-to-a-new-git-repository 

 * http://www.readwriteweb.com/hack/2010/11/how-to-switch-from-svn-to-gith.php
 * https://github.com/nirvdrum/svn2git


recipe for git to svn conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * http://john.albin.net/git/convert-subversion-to-git
 * https://github.com/JohnAlbin/git-svn-migrate


git-svn 
~~~~~~~~~
 
 * git-svn is the canonical approach

     * http://subgit.com/documentation/gitsvn.html claims limited, not a once only transtition than plain and simple git


subgit : keeps both alive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * http://subgit.com/

     * http://subgit.com/documentation/gitsvn.html

Trac issues to Github issues
------------------------------

 * probably need to cook something up  
 * :google:`SVN+Trac to github`
 * http://vincent.bernat.im/en/blog/2011-migrating-to-github.html

       * http://syncwith.us/sd/
       * http://home.gna.org/forgeplucker/


Trac wiki content to Sphinx content
------------------------------------

Trac wiki pages for me somehow turn into write-only content much more  
than the immediacy of plain rst files in your repo, ready to be compiled
by Sphinx into pretty html or PDF.

Need:

   * xmlrpc (or some such) pulling from Trac DB wiki pages into files
   * tracwikitext to (sphinx)rst conversion



XMLRPC access to Trac
-----------------------

 * ~/e/otrac/xmlrpc-wiki-backup.py 
  
    * requires  

