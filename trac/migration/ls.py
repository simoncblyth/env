#!/usr/bin/env python
"""
::

    delta:migration blyth$ ./ls.py ls.py 
    +-----+-----+---+-----------------------------------------------------------------------------------------+
    |   0 |   0 | 0 | #!/usr/bin/env python                                                                   |
    |   1 |   0 | 0 | import sys, codecs, logging                                                             |
    |   2 |   0 | 0 | log = logging.getLogger(__name__)                                                       |
    |   3 |   0 | 0 |                                                                                         |
    |   4 |   0 | 0 | class LS(list):                                                                         |
    |   5 |   4 | 0 |     # Prepares an RST grid table with line indent info for any text.                    |
    |   6 |   4 | 0 |     def __init__(self, text, skips=[]):                                                 |
    |   7 |   8 | 0 |         lines = text.split("\n")                                                        |
    |   8 |   0 | 0 |                                                                                         |
    |   9 |   8 | 0 |         maxlen = max(map(len, lines))                                                   |
    |  10 |   8 | 0 |         wid = (3, 3, 1, maxlen  )                                                       |

"""
import sys, codecs, logging
log = logging.getLogger(__name__)

class LS(list):
    # Prepares an RST grid table with line indent info for any text.
    def __init__(self, text, skips=[]):
        lines = text.split("\n")

        #maxlen = max(map(len, lines))
        maxlen = 100 
        wid = (1, 3, 3, 3, maxlen, 15 )
        fmt = "| %%(skip)%dd | %%(idx)%dd | %%(indent)%dd | %%(offset)%dd | %%(dline)-%ss | %%(kls)-.%ss |" % wid 
        div = "".join(["+-","-+-".join(map(lambda w:"-"*w, wid)), "-+"])

        self._sli = slice(0, len(lines))
        self.div = div
        self.fmt= fmt
        self.skips = skips

        ls = map(lambda _:L(_, self), enumerate(lines))
        list.__init__(self, ls)

    def interleaved(self):
        rows = range(2*len(self[self._sli])+1)
        rows[::2] = [self.div for _ in range(len(self[self._sli])+1)]
        rows[1::2] = map(unicode, self[self._sli])
        return rows

    def bookends(self):
        rows = range(len(self[self._sli])+2)
        rows[0] = self.div 
        rows[1:len(self[self._sli])+1] = map(unicode, self[self._sli])
        rows[len(self[self._sli])+1] = self.div
        return rows

    def __unicode__(self):
        #return "\n".join(self.interleaved())
        return "\n".join(self.bookends())
        
    def __str__(self):
        return unicode(self).encode('utf-8')


       


class L(dict):
    """
    Used to hold the source TracWiki lines, and make initial 
    adjustments to start translation to RST
    """
    def __init__(self, idx_line, ls=None):
        idx, line = idx_line
        indent = len(line) - len(line.lstrip())  # all spaces line gives length of line
        skip = int(ls is not None and line.startswith(tuple(ls.skips)))

        bullet = line.find("* ")
        offset = -indent if indent == bullet else 0   # Trac requires indented bullet lists, which is problematic with RST : so offset 

        dict.__init__(self, idx=idx, orig=line, line=line[indent:], tline=line[:100],  skip=skip, fmt=ls.fmt, indent=indent, kls="..", offset=offset) 
        self.ls = ls


    def _get_dline(self):
        """ with offset 0 iline is same as original line"""
        return " " * self.dindent + self['line'] 
    dline = property(_get_dline)

    def _get_dindent(self):
        """ with offset 0 indent is same as original"""
        return self['indent'] + self['offset']
    dindent = property(_get_dindent)
  
    def __unicode__(self):
        d = self.copy()
        d.update(dline=self.dline)
        return self['fmt'] % d









def test_ls(text):
    ls = LS(text) 
    print ls 


def prep(txt):
    return "\n".join(map(lambda _:_[4:], txt.split("\n")[1:-1]))
 

def test_pre_bullet_spacer():
    """
    hmm dont like inserting extra line, 
    as that makes connection to source linenos problematic 
    hmm: simpler just to duplicate the source reference, then not such a big deal 
    """ 

    text = u"""
    Line before bullet list is invalid in RST, not in TracWiki
     * first bullet
     * second bullet

    """
    
    ls = LS(prep(text))
    print ls

    
def test_arb():
    path = sys.argv[1]
    text = codecs.open(path, encoding='utf-8').read()
    ls = LS(text)
    for l in ls:
        #l['offset'] = -l['indent']
        l['offset'] = 0
    print ls 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #test_arb()
    test_pre_bullet_spacer()
