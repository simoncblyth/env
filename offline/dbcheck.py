import os
from datetime import datetime
from dbtablecounts import DBTableCounts

if __name__=='__main__':

    #group = os.environ.get("DB_GROUP","test")
    #stamp = datetime.strftime( datetime.now() , "%Y%m%d" )
    #stamp = '20100331'
    
    #dbtc = DBTableCounts( group=group , stamp=stamp )
    #print "main "
    #print dbtc

    ## over the persisted instances
    
    #DBTableCounts._summary()
    
    
    a = DBTableCounts( group="client", stamp="20100330")
    b = DBTableCounts( group="local" , stamp="20100330")
    
    
    


    



