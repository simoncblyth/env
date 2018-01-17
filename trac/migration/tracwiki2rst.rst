tracwiki2rst.py
=================

::

   ./tracwiki2rst.py $(wtracdb-path) --onepage 3D


Review Trac wikitext to html conversion
------------------------------------------

Finding Trac
~~~~~~~~~~~~~~~

::

    g4pb:~ blyth$ /usr/bin/python -c "import trac ; print trac.__file__ "
    /usr/local/env/trac/package/tractrac/trac-0.11/trac/__init__.pyc

    g4pb:~ blyth$ trac-
    g4pb:~ blyth$ tractrac-
    g4pb:~ blyth$ tractrac-cd
    g4pb:trac-0.11 blyth$ pwd
    /usr/local/env/trac/package/tractrac/trac-0.11
    g4pb:trac-0.11 blyth$ 

    /usr/local/env/trac/package/tractrac/trac-0.11/trac/wiki/parser.py
    /usr/local/env/trac/package/tractrac/trac-0.11/trac/wiki/formatter.py


Groking the parse
~~~~~~~~~~~~~~~~~~~~

For following wikitext to html note there is some dynamic method calling::

     790     # -- Wiki engine
     791    
     792     def handle_match(self, fullmatch):
     793         for itype, match in fullmatch.groupdict().items():
     794             if match and not itype in self.wikiparser.helper_patterns:
     795                 # Check for preceding escape character '!'
     796                 if match[0] == '!':
     797                     return escape(match[1:])
     798                 if itype in self.wikiparser.external_handlers:
     799                     external_handler = self.wikiparser.external_handlers[itype]
     800                     return external_handler(self, match, fullmatch)
     801                 else:
     802                     internal_handler = getattr(self, '_%s_formatter' % itype)
     803                     return internal_handler(match, fullmatch)
     804 
     805     def replace(self, fullmatch):
     806         """Replace one match with its corresponding expansion"""
     807         replacement = self.handle_match(fullmatch)
     808         if replacement:
     809             return _markup_to_unicode(replacement)
     810 
     ...
     828     def format(self, text, out=None, escape_newlines=False):
     829         self.reset(text, out)
     830         for line in text.splitlines():
     831             # Handle code block
     832             if self.in_code_block or line.strip() == WikiParser.STARTBLOCK:
     833                 self.handle_code_block(line)
     834                 continue
     835             # Handle Horizontal ruler
     836             elif line[0:4] == '----':
     837                 self.close_table()
     838                 self.close_paragraph()
     839                 self.close_indentation()
     840                 self.close_list()
     841                 self.close_def_list()
     842                 self.out.write('<hr />' + os.linesep)
     843                 continue
     844             # Handle new paragraph
     845             elif line == '':
     846                 self.close_paragraph()
     847                 self.close_indentation()
     848                 self.close_list()
     849                 self.close_def_list()
     850                 continue
     851 
     852             # Tab expansion and clear tabstops if no indent
     853             line = line.replace('\t', ' '*8)
     854             if not line.startswith(' '):
     855                 self._tabstops = []
     856 
     857             if escape_newlines:
     858                 line += ' [[BR]]'
     859             self.in_list_item = False
     860             self.in_quote = False
     861             # Throw a bunch of regexps on the problem
     862             result = re.sub(self.wikiparser.rules, self.replace, line)
     863 
     864             if not self.in_list_item:
     865                 self.close_list()
     866 
     867             if not self.in_quote:



re.sub with callable replace
------------------------------

::

    In [3]: re.sub?
    Type:       function
    String Form:<function sub at 0x10f8d38c0>
    File:       /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/re.py
    Definition: re.sub(pattern, repl, string, count=0, flags=0)
    Docstring:
    Return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement repl.  repl can be either a string or a callable;
    if a string, backslash escapes in it are processed.  If it is
    a callable, it's passed the match object and must return
    a replacement string to be used.

::

     805     def replace(self, fullmatch):
     806         """Replace one match with its corresponding expansion"""
     807         replacement = self.handle_match(fullmatch)
     808         if replacement:
     809             return _markup_to_unicode(replacement)
 



wikiparser.rules
------------------

::

    135     def _prepare_rules(self):
    136         from trac.wiki.api import WikiSystem
    137         if not self._compiled_rules:
    138             helpers = []
    139             handlers = {}
    140             syntax = self._pre_rules[:]
    141             i = 0
    142             for resolver in WikiSystem(self.env).syntax_providers:
    143                 for regexp, handler in resolver.get_wiki_syntax():
    144                     handlers['i' + str(i)] = handler
    145                     syntax.append('(?P<i%d>%s)' % (i, regexp))
    146                     i += 1
    147             syntax += self._post_rules[:]
    148             helper_re = re.compile(r'\?P<([a-z\d_]+)>')
    149             for rule in syntax:
    150                 helpers += helper_re.findall(rule)[1:]
    151             rules = re.compile('(?:' + '|'.join(syntax) + ')', re.UNICODE)

    ///   whopper regexp, long list of syntax rules ORed together
    ///

    152             self._external_handlers = handlers
    153             self._helper_patterns = helpers
    154             self._compiled_rules = rules


WikiFormatting
----------------

* http://g4pb.local/tracs/workflow/wiki/WikiFormatting


exercise
----------

::

    In [5]: syntax = [r"(?P<indent>^(?P<idepth>\s+)(?=\S))"]

    In [6]: rules = re.compile('(?:' + '|'.join(syntax) + ')', re.UNICODE)

    In [12]: def replace(fullmatch):
       ....:     if fullmatch is not None:
       ....:         print fullmatch.groupdict()
       ....:     return "dummy"
       ....: 

    In [13]: re.sub(rules, replace, "     hello")
    {'idepth': '     ', 'indent': '     '}
    Out[13]: 'dummyhello'

    In [14]: 

