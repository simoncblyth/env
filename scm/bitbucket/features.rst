Bitbucket Features
====================

Publishing Pages
-----------------

* https://confluence.atlassian.com/display/BITBUCKET/Publishing+a+Website+on+Bitbucket

Argh ``scb-`` is probably not allowed as labels ending with hyphens are not valid in hostnames.

* this, and the desire to use common name across accounts 
  prompted name change Aug 7, 2014 from ``scb-`` to ``simoncblyth``


Valid Hostnames
----------------

* http://en.wikipedia.org/wiki/Hostname

Hostnames are composed of series of labels concatenated with dots, as are all
domain names. For example, "en.wikipedia.org" is a hostname. Each label must be
between 1 and 63 characters long, and the entire hostname (including the
delimiting dots) has a maximum of 255 characters.  

The Internet standards (Request for Comments) for protocols 
mandate that component hostname labels may contain only the ASCII letters 'a' through 'z'
(in a case-insensitive manner), the digits '0' through '9', and the hyphen ('-'). 
The original specification of hostnames in RFC 952, mandated that labels could 
not start with a digit or with a hyphen, **and must not end with a hyphen**. 

However, a subsequent specification (RFC 1123) permitted hostname 
labels to start with digits. 
No other symbols, punctuation characters, or white space are permitted.


Tarball Distribution ?
-------------------------

* https://confluence.atlassian.com/pages/viewpage.action?pageId=273877699


Keep in mind Bitbucket is a code hosting service not a file sharing service.
If a lot of your files are extremely large or if your files are binaries or
executables, you should understand Git or Mercurial will not work well with
them. You'll find that even locally your repository is barely usable. Moreover,
Bitbucket can't display diffs on binaries.

For binary or executable storage, we recommend you look into file hosting
services  such as DropBox, rsync, rsnapshot, rdiff-backup, and so forth.  Still
not sure what to do? Review this post on stackoverflow for more ideas.

* http://stackoverflow.com/questions/1284669/how-do-i-manage-large-art-assets-appropriately-in-dvcs



