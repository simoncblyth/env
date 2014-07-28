hg convert
============

converting env from svn to hg 
--------------------------------

#. empty directories in SVN not honoured in HG ?

   * workaround, delete empties where possible or add README.txt 

#. svn:ignores 


Guides to Conversion
----------------------

* https://code.google.com/p/support/wiki/ConvertingSvnToHg


hg convert documentation
-------------------------

* http://mercurial.selenic.com/wiki/ConvertExtension


filemap to include/exclude/rename
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`--filemap`
    Convert can also filter or rename files during conversion, when you supply it a mapping via the --filemap option.



Splitting Convert
--------------------


* http://hgtip.com/tips/advanced/2009-11-16-using-convert-to-decompose-your-repository/

With a filemap like the below and extract a subfolder into its own repo with its own history::

    include Something 
    rename Something .







