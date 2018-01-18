#!/usr/bin/env python

import re, logging
log = logging.getLogger(__name__)


class InlineTracWiki2RST(object):
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

    _pre_rules = [ 
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

    def __init__(self):
        self._compiled_rules = None

    def _get_rules(self):
        self._prepare_rules()
        return self._compiled_rules
    rules = property(_get_rules)

    def _prepare_rules(self):
        if not self._compiled_rules:
            syntax = self._pre_rules[:]
            rules = re.compile('(?:' + '|'.join(syntax) + ')', re.UNICODE)
            self._compiled_rules = rules

    def __call__(self, line):
        result = re.sub(self.rules, self.replace, line)        
        return result

    def replace(self, fullmatch):
        """Replace one match with its corresponding expansion"""
        replacement = self.handle_match(fullmatch)
        return replacement

    def handle_match(self, fullmatch):
        d = fullmatch.groupdict()
        #print "handle_match:", d

        internal_handler = None

        if d.get('table_cell', None) != None:
            itype, match = "table_cell", d['table_cell']
            internal_handler = getattr(self, '_%s_formatter' % itype)
        elif d.get('last_table_cell', None) != None:
            itype, match = "last_table_cell", d['last_table_cell']
            internal_handler = getattr(self, '_%s_formatter' % itype)
        else:
            for itype, match in d.items():
                if not match:continue
                internal_handler = getattr(self, '_%s_formatter' % itype)
            pass
        pass
        assert internal_handler
        result = internal_handler(match, fullmatch)
        return result


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



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    conv = InlineTracWiki2RST()

    text = r"""

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



    """

    lines = map(lambda _:_.lstrip(), text.strip().split("\n"))

    div = "\n" + "*" * 100 + "\n"

    print div
    print "\n".join(lines)
    print div
    print "\n".join(map(conv, lines))
    print div
  



