#!/usr/bin/env python
    

class Table(list):
    """
    Padding is workaround for non-ascii unicode char widths poking outside RST table
    """
    def __init__(self, *args, **kwa):
        self.pad = kwa.pop("pad", 1)
        list.__init__(self,*args, **kwa)

    def _get_widths(self): 
        wid = map(len, self[0])
        for row in self[1:]:
            w = map(len, row) 
            assert len(wid) == len(w)
            for i,c in enumerate(w):
                wid[i] = max(wid[i], w[i])
            pass
        return wid
    widths = property(_get_widths)

    def make_div(self, mkr):
        return "  ".join(map(lambda n:mkr*n*self.pad, self.widths))

    def make_fmt(self):
        return "  ".join(map(lambda n:"%-"+str(n*self.pad)+"s", self.widths))

    fmt = property(lambda self:self.make_fmt())
    div = property(lambda self:self.make_div("="))
    sep = property(lambda self:self.make_div("-"))

    def __repr__(self):
        return "<Table rows:%d widths:%s pad:%s>" % (len(self),self.widths, self.pad)

    def __unicode__(self):
        div = self.div
        fmt = self.fmt 
        row = [fmt % tuple(r) for r in self]
        return "\n".join([div, row[0], div] + row[1:] +[div])

    def __str__(self):
        return unicode(self).encode("utf-8")



if __name__ == '__main__':

    

    #U = ""
    U = unichr(0xbeef)*10

    a = [u"red", "green", "blue"]
    b = ["1", "2", "3"]
    c = ["cyan", "magenta " + U, "yellow"]

    t = Table([a,b,c], pad=2)

    print t
    print "\n\n"
    t.extend([a,b,c])
    print t

    from env.doc.rstutil import rst2html_open    
    rst2html_open(unicode(t), "t")


