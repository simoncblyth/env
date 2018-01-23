#!/usr/bin/env python
"""

ISSUES
--------

Inline literalization of Image macros not honoured.

* `Image[[path.png]]` 


Anything indented following a literal gets commented, 
real world example.

* http://g4pb.local/tracs/workflow/wiki/ArcNetworkBackup#ObserveOnArc:SetupNetworkBackupforsmallwebshareinArcDSM2.1-0835


"""

import re, logging
log = logging.getLogger(__name__)


from env.doc.tabrst import Table
from env.trac.migration.rsturl  import EscapeURL



class InlineTrac2Sphinx(object):
    """ 
    The ctx.inliner_  canonical instance of this is instanciated on high, 
    up in Trac2Sphinx.make_context, and is used throughout via ctx passing.

    Para.rst
       calls inline() which invokes the inliner_

    SimpleTable.rst
       applys the inliner_ across all table cells
    

    """
    def __init__(self, ctx):
        self.ctx = ctx
        self.eurl = EscapeURL(ctx)
        self.ilnk = InlineTrac2SphinxLink(ctx) 
        self.inli = InlineTracWiki2RST(ctx)
        self.erst = InlineEscapeRST(ctx)

    def __call__(self, enu_line):
        if type(enu_line) is tuple:
            enu, line = enu_line
        else:
            enu, line = -1, enu_line
        pass 
        iline = self.erst(self.ilnk(self.inli(self.eurl(line)))) 

        self.ctx.elem.ind_[enu] = self.ctx.indent
        if iline.strip() != "":
            iline = "%s (%s)(%s)" % (iline, self.ctx.indent, enu)
        pass
        return iline


class ReReplacer(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self._compiled_rules = None
        self._line = None
        self._table = None

    def _get_rules(self):
        self._prepare_rules()
        return self._compiled_rules
    rules = property(_get_rules)

    def _prepare_rules(self):
        if not self._compiled_rules:
            syntax = self._rules[:]
            rules = re.compile('(?:' + '|'.join(syntax) + ')', re.UNICODE)
            self._compiled_rules = rules

    def __call__(self, line):
        self._line = line
        result = re.sub(self.rules, self.replace, line)        
        if result != line: 
            log.debug("[%s]%s" % (self.__class__.__name__, line))
            log.debug("[%s]%s" % (self.__class__.__name__, result))
        pass
        return result

    def replace(self, fullmatch):
        """Replace one match with its corresponding expansion"""
        replacement = self.handle_match(fullmatch)
        return replacement

    def handle_match(self, fullmatch):
        """
        Subclasses may need to override if particular ordering is needed
        """
        d = dict(filter( lambda kv:kv[1] is not None, fullmatch.groupdict().items() ))

        #log.debug("handle_match: %s" %  d)
        internal_handler = None
        for itype, match in d.items():
            internal_handler = getattr(self, '_%s_formatter' % itype)
        pass
        assert internal_handler, ("no handler for %s " % d )
        result = internal_handler(match, fullmatch)
        return result



class InlineEscapeRST(ReReplacer):
    """
    Heuristically avoid unpaired * and ** messing up RST products by escaping them 
    """
    BOLD_TOKEN = "**"
    BOLD_TOKEN_ESCAPE = "\*\*"
    ITALIC_TOKEN = "*"
    ITALIC_TOKEN_ESCAPE = "\*"

    _rules = [ 
        r"(?P<bold>%s)" % BOLD_TOKEN_ESCAPE,
        r"(?P<italic>%s)" % ITALIC_TOKEN_ESCAPE
        ]
 
    _find_bold = re.compile("(?P<findbold>\*{2})")
    _find_italic = re.compile("(?P<finditalic>\*{1})")

    def __init__(self, ctx):
        ReReplacer.__init__(self, ctx)

    def __call__(self, line):
        """
        Override to handle bullet lists
        """
        inset = 0 
        if line.lstrip()[0:2] == "* ":   # guessing a bullet list 
            inset = line.index("*")+1
        pass
        self._line = line[inset:]
        result = re.sub(self.rules, self.replace, line[inset:])        
        return line[:inset] + result

    def _bold_formatter(self, match, fullmatch):
        """
        Escape tokens than do not appear in pairs for the line
        """
        num = len(self._find_bold.findall(self._line))
        if num % 2 == 1:
            replace = self.BOLD_TOKEN_ESCAPE
        else:
            replace = self.BOLD_TOKEN
        pass
        return replace 

    def _italic_formatter(self, match, fullmatch):
        num = len(self._find_italic.findall(self._line))
        if num % 2 == 1:
            replace = self.ITALIC_TOKEN_ESCAPE
        else:
            replace = self.ITALIC_TOKEN
        pass
        return replace 

       

class TableTracWiki2RST(ReReplacer):
    """
    NB this just grabs the cells of the table into a list of lists within Table
    subsequent processing such as inline replacements should happen elsewhere 
    """

    TABLE_TOKEN = "||"
    _rules = [ 
        # || table ||
        r"(?P<last_table_cell>\|\|\s*$)",
        r"(?P<table_cell>\|\|.*?)(?=\|\|)",
        r"(?P<blankline>^$)"
        ]


    def __init__(self, ctx):
        ReReplacer.__init__(self, ctx)

    def handle_match(self, fullmatch):
        d = dict(filter( lambda kv:kv[1] is not None, fullmatch.groupdict().items() ))
        log.debug("handle_match: %s" %  d)
        internal_handler = None
        if d.has_key('table_cell'):
            itype, match = "table_cell", d['table_cell']
            internal_handler = getattr(self, '_%s_formatter' % itype)
        elif d.has_key('last_table_cell'):
            itype, match = "last_table_cell", d['last_table_cell']
            internal_handler = getattr(self, '_%s_formatter' % itype)
        else:
            for itype, match in d.items():
                internal_handler = getattr(self, '_%s_formatter' % itype)
            pass
        pass
        assert internal_handler, ("no handler for %s " % d )
        result = internal_handler(match, fullmatch)
        return result

    def _last_table_cell_formatter(self, match, fullmatch):
        return self.do_table_cell(match, fullmatch, last=True)

    def _table_cell_formatter(self, match, fullmatch):
        return self.do_table_cell(match, fullmatch, last=False)

    def do_table_cell(self, match, fullmatch, last=False):
        if self._table is None:
            self._table = Table()
            self._row = []
        pass
        if last:
            self._table.append(self._row)
            self._row = [] 
        else:
            content = match[2:]
            self._row.append(content)
            log.debug("do_table_cell match:%s content:%s " % (match, content))
        pass
        return ''

    def _blankline_formatter(self, match, fullmatch):
        if self._table is not None:
            replacement = "\n"+unicode(self._table) 
            self._table = None
        else:
            replacement = ''
        pass
        return replacement
 



def test_InlineTrac2SphinxLink():
    """
    use ":set list" in vim to see the trailing whitespace 
    that is also required to match
    """
    text = u"""
    * source:/trunk/pyrex/main/Makefile
    * source:trunk/pyrex/main/Makefile 
    * source:trunk/autosurvey/autosurvey.py 
    * source:trunk/python/ipython.bash@24
    * source:trunk/autosurvey/autosurvey.py source:trunk/python/ipython.bash

    * env:/trunk/base/cron.bash

    * wiki:LXML
    * htdocs:db2trac.xsl
    * NTU:something

    * google:"osx install kext"
    * google:"rsync.plist"

    * http://www.google.com
    * https://www.google.com
    * file:///tmp/dummy.txt
    * x-man-page://osascript
    * smb://user@server/test
    * afp://user@server/test

    * 14:00

    """

    x_text = u"""
    * :source:`/trunk/pyrex/main/Makefile`
    * :source:`trunk/pyrex/main/Makefile` 
    * :source:`trunk/autosurvey/autosurvey.py` 
    * :source:`trunk/python/ipython.bash@24`
    * :source:`trunk/autosurvey/autosurvey.py` :source:`trunk/python/ipython.bash`

    * :env:`/trunk/base/cron.bash`

    * :wiki:`LXML`
    * :htdocs:`db2trac.xsl`
    * :ntu:`something`

    * :google:`osx install kext`
    * :google:`rsync.plist`

    * http://www.google.com
    * https://www.google.com
    * file:///tmp/dummy.txt
    * x-man-page://osascript
    * smb://user@server/test
    * afp://user@server/test

    * 14:00

    """
    return test_translate( InlineTrac2SphinxLink, text, x_text )

 
class InlineTrac2SphinxLink(ReReplacer):
    """
    Doing a replacement that can be switched off with a preceeding escape character, 
    easiest to maybe capture the escape and put the logic into the replace callable::

        In [30]: re.sub(re.compile('(?:(?P<the_a>[!`]?a)|(?P<the_b>b))', re.UNICODE), lambda m:"["+m.group(0)+"]", " some !ab line ab `a b ")
        Out[30]: ' some [!a][b] line [a][b] [`a] [b] '

    """
    EXCLUDE = ["http", "https", "file", "x-man-page", "smb", "afp", "mail", "ftp", "xmpp", "svn", "git", "message" ]
    TRAC_EXCLUDE = ["milestone", "query", "comment", "changeset", "log", "diff", "attachment", "tagged"]
    EXCLUDE += TRAC_EXCLUDE


    LINKTYPE = "[a-z-]+"   # numbers and underscore are not permissible in RST role types 

    LINK_TOKEN = ":"
    ESCAPED_LINK_TOKEN = "\:"
    
    DISALLOW_CHARS = "!`*"

    _rules = [ 
       r"(?P<link>[%s]?%s%s[\w\/]\S+)" % (DISALLOW_CHARS, LINKTYPE, ESCAPED_LINK_TOKEN) ,      # 2nd \w disallows quote after colon
       r"(?P<qlink>[%s]?\w+%s\"[\S ]+\")" % (DISALLOW_CHARS, ESCAPED_LINK_TOKEN ), 
           ]

    def __init__(self, ctx):
        ReReplacer.__init__(self, ctx)



    def _escaped(self, match):
        escaped = match[0] in self.DISALLOW_CHARS 
        if escaped:
            if match[0] == "!":
                return match[1:]
            else:
                return match
            pass
        else:
            return None
        pass

    def _qlink_formatter(self, match, fullmatch):
        tlnk = fullmatch.group('qlink')
        assert tlnk == match 
        escaped = self._escaped(match)
        if escaped is not None:
            return escaped
        pass        

        f = tlnk.index(self.LINK_TOKEN)
        typ = tlnk[:f]
        arg = tlnk[f+1:]
        assert arg[0] == '"' and arg[-1] == '"'
        uarg = arg[1:-1]
        xlnk = ":%s:`%s`" % (typ, uarg)
        #log.debug("(qlink) tlnk:[%(tlnk)s] typ:[%(typ)s] arg:[%(arg)s] uarg:[%(uarg)s] xlnk:[%(xlnk)s]" % locals())
        return xlnk

    def _link_formatter(self, match, fullmatch):
        tlnk = fullmatch.group('link')
        assert tlnk == match 
        escaped = self._escaped(match)
        if escaped is not None:
            return escaped
        pass        

        f = tlnk.index(self.LINK_TOKEN)
        typ = tlnk[:f]
        exclude = typ in self.EXCLUDE 
        if exclude:
            return match
        pass

        arg = tlnk[f+1:]
        xlnk = ":%s:`%s`" % (typ, arg)

        #log.debug("(link) tlnk:[%(tlnk)s] typ:[%(typ)s] arg:[%(arg)s] xlnk:[%(xlnk)s]" % locals())
        return xlnk 

 

class InlineTracWiki2RST(ReReplacer):
    """
    Using extracts from Trac 0.11 WikiParser to facilitate wikitxt interpretation 

    g4pb:/usr/local/env/trac/package/tractrac/trac-0.11/trac/wiki/parser.py

    http://g4pb.local/tracs/workflow/wiki/WikiFormatting

    http://docutils.sourceforge.net/docs/user/rst/quickref.html#inline-markup 

    """

   # Some constants used for clarifying the Wiki regexps:

    BOLDITALIC_TOKEN = "'''''"
    BOLD_TOKEN = "'''"
    ITALIC_TOKEN = "''"
    UNDERLINE_TOKEN = "__"
    STRIKE_TOKEN = "~~"
    SUBSCRIPT_TOKEN = ",,"
    SUPERSCRIPT_TOKEN = r"\^"
    INLINE_TOKEN = "`" 
    STARTBLOCK_TOKEN = r"\{\{\{"
    STARTBLOCK = "{{{"
    ENDBLOCK_TOKEN = r"\}\}\}"
    ENDBLOCK = "}}}"
    
    LINK_SCHEME = r"[\w.+-]+" # as per RFC 2396
    INTERTRAC_SCHEME = r"[a-zA-Z.+-]*?" # no digits (support for shorthand links)

    QUOTED_STRING = r"'[^']+'|\"[^\"]+\""

    SHREF_TARGET_FIRST = r"[\w/?!#@](?<!_)" # we don't want "_"
    SHREF_TARGET_MIDDLE = r"(?:\|(?=[^|\s])|[^|<>\s])"
    SHREF_TARGET_LAST = r"[\w/=](?<!_)" # we don't want "_"

    LHREF_RELATIVE_TARGET = r"[/#][^\s\]]*|\.\.?(?:[/#][^\s\]]*)?"

    XML_NAME = r"[\w:](?<!\d)[\w:.-]*?" # See http://www.w3.org/TR/REC-xml/#id 

    # Sequence of regexps used by the engine

    #    (?:...)  Non-grouping version of regular parentheses.
    #    (?P<name>...) The substring matched by the group is accessible by name.
    #    (?P=name)     Matches the text matched earlier by the group named name.

    _rules = [ 
        r"(?P<indent>^(?:\s+)(?=\S))",
        # Font styles
        r"(?P<bolditalic>!?%s)" % BOLDITALIC_TOKEN,
        r"(?P<boldpair>!?%s(?:.*?)%s)" % ( BOLD_TOKEN, BOLD_TOKEN ),
        r"(?P<bold>!?%s)" % BOLD_TOKEN,
        r"(?P<italicpair>!?%s(?:.*?)%s)" % ( ITALIC_TOKEN, ITALIC_TOKEN ),
        r"(?P<italic>!?%s)" % ITALIC_TOKEN,
        r"(?P<underline>!?%s)" % UNDERLINE_TOKEN,
        r"(?P<strike>!?%s)" % STRIKE_TOKEN,
        r"(?P<subscript>!?%s)" % SUBSCRIPT_TOKEN,
        r"(?P<superscript>!?%s)" % SUPERSCRIPT_TOKEN,
        r"(?P<inlinecode>!?%s(?:.*?)%s)" \
        % (STARTBLOCK_TOKEN, ENDBLOCK_TOKEN),
        r"(?P<inlinecode2>!?%s(?:.*?)%s)" \
        % (INLINE_TOKEN, INLINE_TOKEN)]

      
    def __init__(self, ctx):
        ReReplacer.__init__(self, ctx)

    def _inlinecode_formatter(self, match, fullmatch):
        l = len(self.STARTBLOCK)
        code = fullmatch.group('inlinecode')[l:-l]
        return "``%s``" % code.strip()

    def _inlinecode2_formatter(self, match, fullmatch):
        l = len(self.INLINE_TOKEN)
        code = fullmatch.group('inlinecode2')[l:-l]
        return "``%s``" % code.strip()

    def _bold_formatter(self, match, fullmatch):
        return "**"

    def _italic_formatter(self, match, fullmatch):
        return "*"

    def _bolditalic_formatter(self, match, fullmatch):
        return "**"

    def _underline_formatter(self, match, fullmatch):
        return "*"

    def _superscript_formatter(self, match, fullmatch):
        return " TODO-superscript-formatter "

    def _subscript_formatter(self, match, fullmatch):
        return " TODO-subscript-formatter "

    def _strike_formatter(self, match, fullmatch):
        return " TODO-strike-formatter "

    def _boldpair_formatter(self, match, fullmatch):
        l = len(self.BOLD_TOKEN)
        c = fullmatch.group('boldpair')[l:-l]
        return "**%s** " % c.strip()

    def _italicpair_formatter(self, match, fullmatch):
        l = len(self.ITALIC_TOKEN)
        c = fullmatch.group('italicpair')[l:-l]
        return "*%s* " % c.strip()

    def _indent_formatter(self, match, fullmatch):
        indent = fullmatch.group('indent')
        self.ctx.indent = len(indent)
        return indent



def unindent_split_(text):
    return map(lambda _:_[4:], text.split("\n")[1:-1])

def test_translate( cls, text, x_text=None ):

    ctx = {}
    translator = cls(ctx) 
    not_None_ = lambda _:_ is not None  # avoid blanks lines from table rows

    lines = unindent_split_(text)
    r_lines = filter(not_None_,map(translator, lines))   

    rst = "\n".join(r_lines)
    div = "\n" + "*" * 100 + "\n"
    print div
    print "\n".join(lines)
    print div
    print rst
    print div


    if x_text is not None:
        x_lines = unindent_split_(x_text)
        dif = 0
        for i, (rl, xl) in enumerate(zip(r_lines, x_lines)):
            if rl != xl:
                dif += 1
                log.warning(" unexpected translation at line %s " % i )
                print "r: [%s]" % rl
                print "x: [%s]" % xl
            pass
        pass
        log.info("line by line comparison dif:%s r:%s x:%s " % (dif, len(r_lines), len(x_lines)))
        pass
        x_rst = "\n".join(x_lines)
        assert x_rst == rst 
    pass

    return rst 
 

def test_InlineEscapeRST():
    text = r"""

    * line with a correct *italic-1*  and *italic-2*
    * line with marker needing escape *nix 

    * line with a correct **bold**  and *italic*
    * line with marker needing escape **nix 

    """
    return test_translate( InlineEscapeRST, text )

def test_TableTracWiki2RST():
    text = r"""
    = test_TableTracWiki2RST =

    Text on line before table
    || red || green || blue ||
    || red || green || blue ||
    || red || green || blue ||
    || red || green || blue ||
    || red || green || blue ||

    || '''red''' || green || blue ||
    ||    red    || green || blue ||
    ||    red    || green || blue ||
    ||    red    || green || blue ||
    ||    red    || green || blue ||

    """
    return test_translate( TableTracWiki2RST, text )


def test_InlineTracWiki2RST():
    text = r"""
    = test_InlineTracWiki2RST =

    * headings are not handled inline (yet)

    * inlinecode blocks {{{cron-*}}} via triple curlies 
    * inlinecode2 blocks with `backticks`
    * '''bold'''
    * ''italic'' 

    Inline markup not simply supported in RST with Replacements

    * '''''bolditalic''''' to '''''bold'''''  
    * __underline__ to __italic__ 
    * ~~strike-through~~ to ~~bold~~
    * ^superscript^ 
    * ,,subscript,,

    * ''' Bold blocks starting/ending with one or more spaces  '''
    * '' italic blocks starting/ending with one or more spaces  ''

    * '''Item'''s flush to text 

       * indented


    * hmm to support expansion inside cells need to retain 
      the table rather than collapsing down to rst ?

      * see how often this is used

    """
    return test_translate( InlineTracWiki2RST, text )
 



if __name__ == '__main__':
    level = 'DEBUG'
    #level = 'INFO'
    logging.basicConfig(level=getattr(logging, level), format="%(name)s %(lineno)s %(message)s")

    test_InlineTracWiki2RST()
    test_InlineEscapeRST()    
    rst = test_TableTracWiki2RST()
    rst = test_InlineTrac2SphinxLink()

    from env.doc.rstutil import rst2html_open    
    assert type(rst) is unicode
    rst2html_open(rst, "pg")




