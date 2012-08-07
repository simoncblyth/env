
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


if __name__ == '__main__':

   td = TabularData()
   td.append( {"color":"red",   "number":1 }) 
   td.append( {"color":"green", "number":2 }) 
   td.append( {"color":"blue",  "number":3 }) 
   td.append_( color="blue", number="3" ) 
   print td
   print repr(td)



