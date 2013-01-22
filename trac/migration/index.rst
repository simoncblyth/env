Migrating Trac Contents Elsewhere
===================================

What needs to be migrated ? Where to ? What can be dropped ?

* code in SVN
* history of changes in SVN
* issues in Trac 
* wiki pages in Trac (to RST pages for Sphinx consumption?)  

   * translation of macros ?
   * propagation of tags


Objective
-----------

* Keep repositories alive without having to support them, back them up etc..

Problems
---------

* preserve ticket numbers ?
* user mapping ?


What to migrate to 
--------------------

Advantage of github and gitorious are the free for open source services.


redmine
~~~~~~~~

  * http://blog.hsatac.net/2012/01/redmine-migrate-from-trac-0-dot-12/

github
~~~~~~~

  * no server installation possible
  * simple API
  * very popular


gitorious
~~~~~~~~~~


* http://gitorious.org/

   * can install the code for the server, in order to learn the details

* http://getgitorious.com/

   * http://getgitorious.com/documentation/index.html   




Search
--------

* google:"migrate from Trac to github"

http://vincent.bernat.im/en/blog/2011-migrating-to-github.html


::

    At last, I have done it manually. GitHub API is well documented and there
    exists bindings in various languages including Python but it is a very limited
    API. You can?t choose the number of the ticket nor its date.





Tools
------




* https://github.com/adamcik/github-trac-ticket-import 

   * simple script, Trac CSV report into github API calls




