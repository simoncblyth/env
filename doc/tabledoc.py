#!/usr/bin/env python

from copy import copy

try:
    from converter.tabular import TabularData
except ImportError:
    class TabularData(list):
        """
        pale imitation of the real TabularData
	"""
	def as_rst(self, cols ):
	    def cell(v):
		return "/".join(map(lambda _:str(int(_)),v)) if type(v) in (list,tuple) else str(v)
	    def fallback(d):
		return " ".join( map(lambda k:"%-10s" % cell(d[k]), cols ) )
	    return "\n".join(fallback(d) for d in self) + "\n\n"



class AnnotatedTabularData(object):
    """
    Adds annotating roles to TabularData, signalled by cells containing dicts
    with keys v and st where v is the value and st contains one of "alarm", "warn" or "ok"
    for example::

         {"color":"green", "number":dict(v=2,st='alarm') }


    """
    anno_ = dict(
              ok=lambda _:r":ok:`%s`" % _,
           alarm=lambda _:r":alarm:`%s`" % _,
            warn=lambda _:r":warn:`%s`" % _,
           )

    def __init__(self, *args, **kwa):
        self.td = TabularData(*args, **kwa)

    def append(self, *args, **kwa):
        self.td.append(*args, **kwa)

    def append_(self, *args, **kwa):
        self.td.append_(*args, **kwa)

    def copy(self):
        dl = [] 
        for d in self.td:
            dl.append(d)
        return AnnotatedTabularData(dl)

    def _annodict(self, d_, copy_=False):
        """
        Delve into the dict demoting annotation dicts down 
        to annotated strings

        :param d_: dict 
        :param copy_:  operate on a copy rather than inplace
        """
        d = copy(d_) if copy_ else d_
        for k,v in d.items():
            if isinstance(v, dict) and v.has_key('v') and v.has_key('st'):
                vv = v['v']
                vst = v['st']
                d[k] = self.anno_[vst](vv)
        return d 

    def as_rst(self, cols=None, annonly=False):
        """
        :param cols: specify column order rather than default of accepting random dict ordering
        :param annonly:  when True show only rows with annotations

	Seems that docutils is converting backticks into ordinary quotes 
        so diddle it at string level

        For `not(c == d)` there is annotation in the row thus when `annonly=True` the row
        is included in the output.
        """
        cl = []
        for d in self.td:
            c = self._annodict(d, copy_=True)
            if not(c == d and annonly):
                cl.append(c)

        td = TabularData(cl)
        if len(td) == 0:     # avoid rst build warnings from empty table
            return ""
        rst = td.as_rst( cols )
        return rst.replace("'","`") 



if __name__ == '__main__':

   td = AnnotatedTabularData()
   td.append( {"color":"red",   "number":1 }) 
   td.append( {"color":"green", "number":2 }) 
   td.append( {"color":"green", "number":dict(v=2,st='alarm') }) 
   td.append( {"color":"blue",  "number":3 }) 
   td.append_( color="blue", number="3" ) 

   print td
   print "td", td.as_rst()

   tc = td.copy()
   print "tc", tc.as_rst()

   print "tc", td.as_rst(annonly=True)


