#!/usr/bin/env python
    

class Table(list):
    """
    Preparation of RST Simple Tables from a list of lists 

    Padding is workaround for non-ascii unicode char widths poking outside RST table
    """
    def __init__(self, *args, **kwa):
        self.pad = kwa.pop("pad", 1)
        self.hdr = kwa.pop("hdr", False)
        self.kwa = kwa
        self._extracolumn = False
        list.__init__(self,*args, **kwa)

    def _get_widths(self): 
        wid = map(len, self[0])
        for row in self[1:]:
            w = map(len, row) 
            assert len(wid) == len(w), ( len(wid), len(w), repr(self), ",".join(row) ) 
            for i,c in enumerate(w):
                wid[i] = max(wid[i], w[i])
            pass
        pass
        self._extracolumn = len(wid) == 1   
        return wid if not self._extracolumn else wid + [3]
    widths = property(_get_widths)

    def make_div(self, mkr):
        return "  ".join(map(lambda n:mkr*n*self.pad, self.widths))

    def make_fmt(self):
        return "  ".join(map(lambda n:"%-"+str(n*self.pad)+"s", self.widths))

    fmt = property(lambda self:self.make_fmt())
    div = property(lambda self:self.make_div("="))
    sep = property(lambda self:self.make_div("-"))

    def apply_func(self, func_):
        if func_ is None:return
        for ir in range(len(self)):
            for ic in range(len(self[ir])):
                self[ir][ic] = func_(self[ir][ic]) 
            pass
        pass

    def __repr__(self):
        return "<Table rows:%d pad:%s row0:%s  extracolumn:%s>" % (len(self), self.pad, ",".join(self[0]), self._extracolumn )

    def __unicode__(self):
        """
        #. RST doesnt allow single column table, but Trac does : so add an extra blank column 
        #. blank first columns of RST simple tables need to be marked with an empty comment ".."
        
           * http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#id60

        """
        div = self.div
        fmt = self.fmt 

        for r in self:
            if r[0].strip() == "":
                r[0] = ".." 
            pass
        pass   

        if not self._extracolumn:
            rows = [fmt % tuple(r) for r in self]
        else:
            rows = [fmt % tuple(r+[""]) for r in self]
        pass

        if self.hdr:
            body = [rows[0],div] + rows[1:] 
        else:
            body = rows
        pass

        comment = ["", "..", "   table rows:%s widths:%s " % (len(self), self.widths) , "" ]


        return "\n".join(["",div] + body + [div,""] + comment )

    def __str__(self):
        return unicode(self).encode("utf-8")



if __name__ == '__main__':

    

    #U = ""
    U = unichr(0xbeef)*10

    a = [u"red", "green", "blue"]
    b = ["1", "2", "3"]
    c = ["cyan", "magenta " + U, "yellow"]

    t = Table([a,b,c], pad=2, hdr=True)

    print t
    print "\n\n"
    t.extend([a,b,c])
    print t

    from env.doc.rstutil import rst2html_open    
    rst2html_open(unicode(t), "t")


