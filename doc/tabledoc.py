from converter.tabular import TabularData

if __name__ == '__main__':

   td = TabularData()
   td.append( {"color":"red",   "number":1 }) 
   td.append( {"color":"green", "number":2 }) 
   td.append( {"color":"blue",  "number":3 }) 
   td.append_( color="blue", number="3" ) 
   print td
   print repr(td)


