#!/usr/bin/env python
"""
tracwikilite.py
=================

See 

* :doc:`tracwiki2rst.rst`
* g4pb:/usr/local/env/trac/package/tractrac/trac-0.11/trac/wiki/parser.py
* g4pb:/usr/local/env/trac/package/tractrac/trac-0.11/trac/wiki/formatter.py


/Users/blyth/workflow/bin/tab.py


"""
import logging, sys, re, os
from StringIO import StringIO

log = logging.getLogger(__name__)

from env.doc.tabrst import Table

#from env.sqlite.db import DB


class TracWikiLite(object):
    def __init__(self):
        self._compiled_rules = None
        self._line = None

    def _prepare_rules(self):
        """
        (?:...)  Non-grouping version of regular parentheses.
        (?P<name>...) The substring matched by the group is accessible by name.
        (?P=name)     Matches the text matched earlier by the group named name.

        """
        if not self._compiled_rules: 
            syntax = [
                       r"(?P<indent>^(?P<idepth>\s+)(?=\S))",
                       # || table ||
                       r"(?P<last_table_cell>\|\|\s*$)",
                       r"(?P<table_cell>\|\|.*?)(?=\|\|)"
                     ]

            ptn = '(?:' + '|'.join(syntax) + ')'
            print ptn
            rules = re.compile(ptn, re.UNICODE)
            self._compiled_rules = rules 
        pass

    def _get_rules(self):
        self._prepare_rules()
        return self._compiled_rules
    rules = property(_get_rules)


    def reset(self, source, out=None):
        self.source = source
        class NullOut(object):
            def write(self, data): pass 
        self.out = out or NullOut()
        self._open_tags = [] 
        self._list_stack = [] 
        self._quote_stack = [] 
        self._tabstops = [] 

        self.in_code_block = 0
        self.in_table = 0
        self._table = None
        self._row = None

        self.in_def_list = 0
        self.in_table_row = 0
        self.in_table_cell = 0
        self.paragraph_open = 0

        log.info("reset")    

 
    def open_table(self):
        log.info("open_table")    
        if not self.in_table:
            #self.close_paragraph()
            #self.close_list()
            #self.close_def_list()
            self.in_table = 1
            self._table = Table()
            self.out.write('<table class="wiki">' + os.linesep)

    def open_table_row(self):
        log.info("open_table_row")    
        if not self.in_table_row:
            self.open_table()
            self.in_table_row = 1
            self._row = []
            self.out.write('<tr>\n')

    def close_table_row(self):
        log.info("close_table_row")    
        if self.in_table_row:
            self.in_table_row = 0

            if self.in_table_cell:
                self.in_table_cell = 0
                self.out.write('</td>')
            pass
            self.out.write('\n</tr>')

    def close_table(self):
        log.info("close_table")    
        if self.in_table:
            self.close_table_row()
            self.out.write('\n</table>' + os.linesep)
            self.in_table = 0

    def _indent_formatter(self, match, fullmatch):
        idepth = len(fullmatch.group('idepth'))
        log.info("idepth:%s" % idepth) 
        return ''

    def do_table_cell(self, match, fullmatch, last=False):
        self.open_table()
        self.open_table_row()

        if not last:
            content = match[2:]
            self._row.append(content)
            print "do_table_cell match:%s content:%s " % (match, content)
        else:
            self._table.append(self._row)
            self._row = [] 
        pass

        if self.in_table_cell:
            return '</td><td>' 
        else:
            self.in_table_cell = 1
            return '<td>'

    def _last_table_cell_formatter(self, match, fullmatch):
        return self.do_table_cell(match, fullmatch, last=True)

    def _table_cell_formatter(self, match, fullmatch):
        return self.do_table_cell(match, fullmatch, last=False)


    def handle_match(self, fullmatch):
        print "handle_match:", fullmatch.groupdict()

        d = fullmatch.groupdict()

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

    def replace(self, fullmatch):
        replacement = self.handle_match(fullmatch)
        return replacement

    def format(self, text, out=None):
        self.reset(text, out) 
        for line in text.splitlines():
            print "line:%s " % line 
            self._line = line
            result = re.sub(self.rules, self.replace, line) 
            self.out.write(result + os.linesep)

            if self.in_table and line.strip()[0:2] != '||':
                self.close_table()

        pass





if __name__ == '__main__':

    wikitext = r"""

||Cell 1||Cell 2||Cell 3||
||Cell 4||Cell 5||Cell 6||

"""

    out = StringIO()

    twl = TracWikiLite()
    twl.format(wikitext, out )

    div = "~" * 100    

    print div
    print wikitext 
    print div
    print out.getvalue()
    print div

    open("/tmp/t.html", "w").write(out.getvalue())


    print twl._table



