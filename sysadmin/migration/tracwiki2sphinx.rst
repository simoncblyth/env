Conversion of Trac wiki to Sphinx rst
=======================================
   
.. contents:: :local:


Status of TracWiki2Sphinx
----------------------------

Made start (copy of the Trac2MediaWiki) plugin:

* http://dayabay.phys.ntu.edu.tw/tracs/tracdev/browser/tracwiki2sphinx/trunk/tracwiki2sphinx/converter.py
* http://localhost/tracs/workflow/wiki/TracWiki2Sphinx

::

    trac-
    tracwiki2sphinx-
    tracwiki2sphinx-vi 


establish development cycle without the webserver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DONE with standalone MediaWiki converter

* http://dayabay.phys.ntu.edu.tw/tracs/tracdev/changeset/126


Strategy
----------

Approaches:

  * http://dayabay.phys.ntu.edu.tw/tracs/tracdev/browser/trac2mediawiki/trunk/0.11/plugins/trac2mediawiki/converter.py

     * adapt something like this to generate rst, it deals in trac APIs for converters and formatters 
       so offers some higher lever wins over bare regexp handling, at expense of getting intimate with Trac API  
     * the dependence on trac makes for more complicated development and testing : knocked this for six with standalone converter
     * XMLRPC access not so useful here, depends on impinging on the Trac conversion machinery that normally creats
       html from wikitext, and instead creates rst 

  * http://pypi.python.org/pypi/trac2rst

      * quick and dirty regexp based
      * this nicely fits with XMLRPC access to the wikitext   
      * https://bitbucket.org/pcaro/trac2rst/src/9d1b605ac030/src/trac2rst/core.py  
      * would need to add support for my common trac wiki usage patterns  


Trac Formatter based
----------------------

Checkout the trac source

:: 
    trac-
    tractrac-
    tractrac-cd


Formatters such as Trac2MediaWiki work by extending the trac.wiki.formatter.Formatter 
with a large-ish number of specialization formatter methods.
Crucial part of `trac.wiki.formatter.Formatter` subclasses, are the internal handlers with method signature convention `_<type>_formatter(match, fullmatch)`.
The formatter methods as invoked by the `handle_match` method.

::

   # -- Wiki engine
    
    def handle_match(self, fullmatch):
        for itype, match in fullmatch.groupdict().items():
            if match and not itype in self.wikiparser.helper_patterns:
                # Check for preceding escape character '!'
                if match[0] == '!':
                    return escape(match[1:])
                if itype in self.wikiparser.external_handlers:
                    external_handler = self.wikiparser.external_handlers[itype]
                    return external_handler(self, match, fullmatch)
                else:
                    internal_handler = getattr(self, '_%s_formatter' % itype)
                    return internal_handler(match, fullmatch)

The formatter is plugged into trac component system via::

    class Trac2MediaWikiPlugin(Component):
        implements(IContentConverter)







Metadata preservation
--------------------------

Need to find Sphinx/RST equivalent representation of Trac metadata, and preserve this in migration:

#. modification time stamps 

   #. migration issue only? 

#. trac tags 
#. tag lists (not really like toctree)


Sphinx access to docinfo metadata
-----------------------------------

https://groups.google.com/forum/?fromgroups#!topic/sphinx-dev/Hszmw8clhrM

Investigate in ``env.sphinxext.taglist`` looks like only specific metadata keys are read


Trac Macros that I use 
-----------------------

TracNav(ReponameNav) : sidebar navigation panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not really equivalent to the Sphinx toctree: 

  * multi document index, not needed at single doc level with Sphinx
  * will need manual organization as Trac is effectively flat and Sphinx uses a tree 
    
PageOutline
~~~~~~~~~~~~

Inpage content list, translate to::

   .. contents:: :local:

TagCloud
~~~~~~~~~

Only a few of these

TagList + per-page tag listing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need a Sphinx extension to handle this, maybe use standard top of doc fieldlist:

::

    :tags: Red, Green, Blue

And/or extension (assuming can grab the page context and get the metadats tags on that page with local)

::

     .. taglist:: :local:


  * http://sphinx.pocoo.org/markup/misc.html

Trac taglists are dynamic and provide links to pages featuring those tags, 

  * how to do that possible in Sphinx ?



