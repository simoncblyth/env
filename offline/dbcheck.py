import os
from datetime import datetime
from dbtablecounts import DBTableCounts

if __name__=='__main__':

    sect = os.environ.get("DB_SECT","local")
    #stamp = datetime.strftime( datetime.now() , "%Y%m%d" )
    stamp = '20100331'
    
    dbtc = DBTableCounts( sect=sect , stamp=stamp )
    print "main "
    print dbtc

    ## over the persisted instances
    #for i in DBTableCounts._instances():
    #    print i


    



