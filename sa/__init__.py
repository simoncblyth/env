"""
    from env.sa import * 

    Introspect classes from DATABASE_URL and join assuming DBI pairing 
    and bring into this scope 

    CalibPmtSpecVld.count() == 11

"""
from dbi import DbiSoup
from private import Private
p = Private()
dbis = DbiSoup( p('DATABASE_URL') )
locals().update(dbis)              ## populates scope with the mapped classes 


