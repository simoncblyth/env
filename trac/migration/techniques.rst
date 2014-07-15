Techniques
===========


Objectives
-----------

* Keep repositories alive without having to support them, backup etc..

Intro
-------

What needs to be migrated ? Where to ? What can be dropped ?

* code in SVN
* history of changes in SVN
* issues in Trac 

  * https://github.com/acdha/migrate-trac-issues-to-github/blob/master/migrate.py

* wiki pages in Trac into RST Sphinx pages

   * convert to RST pages within the repo, ready for Sphinx consumption?  
   * translation of Trac macros into Sphinx RST equivalents
   * propagation of tags into RST docinfo metadata 
   * advantage: documentation/notes format decoupled from hosting (eg github wiki format markdown?) 
   * disadvantage: no-dynamic wiki editing, html output must wait for repo commits and documentation rebuilds


Need to merge
---------------

* http://dayabay.ihep.ac.cn/e/sysadmin/migration/tracwiki2sphinx/
* http://dayabay.ihep.ac.cn/e/sysadmin/migration/trac2github/


Choices
--------

 * ease of migration comes down in large part to the depth of the API provided


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

bitbucket
~~~~~~~~~~

 * examples

   * https://bitbucket.org/birkenfeld/sphinx/

 * mercurial or git hosting
 * https://confluence.atlassian.com/display/BITBUCKET/bitbucket+Documentation+Home
 * https://confluence.atlassian.com/display/BITBUCKET/Using+a+Bitbucket+Wiki

    * Bitbucket wikis support Creole, Markdown, reStructuredTextte, and Textile syntax.   There is special Bitbucket markup that you can use to link to Bitbucket objects from wikis.

 * API

    * https://confluence.atlassian.com/display/BITBUCKET/Using+the+Bitbucket+REST+APIs



github
~~~~~~~

  * no server installation possible
  * simple API
  
    * http://developer.github.com/


  * markup

     * https://github.com/github/markup
     * http://github.github.com/github-flavored-markdown/


  * very popular

  * http://stackoverflow.com/questions/6671584/how-to-export-trac-to-github-issues

      * https://github.com/trustmaster/trac2github  

          * yuck PHP, assumes Trac is based off mysql 

      * python control of Github API
 
          * http://pypi.python.org/pypi/PyGithub/  
          * http://vincent-jacques.net/PyGithub/  claims full Github API v3 covered

      * from CSV created by Trac SQL queries into github using PyGithub

          * http://pypi.python.org/pypi/tratihubis/
          * https://github.com/roskakori/tratihubis
          * https://github.com/roskakori/tratihubis/blob/master/tratihubis.py looks quite mature, but no markup conversion


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



