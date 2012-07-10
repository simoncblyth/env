Conversion of Trac wiki to Sphinx rst
=======================================

Approaches:

  * http://dayabay.phys.ntu.edu.tw/tracs/tracdev/browser/trac2mediawiki/trunk/0.11/plugins/trac2mediawiki/converter.py

     * adapt something like this to generate rst, it deals in trac APIs for converters and formatters 
       so offers some higher lever wins over bare regexp handling, at expense of getting intimate with Trac API  
     * the dependence on trac makes for more complicated development and testing
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

Crucial part of ``trac.wiki.formatter``, are the internal handlers with method signature convention ``_<type>_formatter(match, fullmatch)``

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



Metadata preservation
--------------------------

Need to find Sphinx/RST equivalent representation of Trac metadata, and preserve this in migration:

#. modification time stamps 
#. trac tags 
#. tag lists (not really like toctree)




