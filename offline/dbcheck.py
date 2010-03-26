from datetime import datetime
from dbtablecounts import DBTableCounts



def today():
    """ create a new instance for today if not existing already  """
    sect = os.environ.get("DB_SECT","local")
    stamp = datetime.strftime( datetime.now() , "%Y%m%d" )
    
    dbtc = DBTableCounts( sect=sect, stamp=stamp )
    print dbtc


if __name__=='__main__':

    #dbtc = DBTableCounts( sect='testdb' , stamp='20100325' )
    dbtc = DBTableCounts( sect='testdb' , stamp='20100330' )
    print dbtc


    ## over the persisted instances
    #for i in DBTableCounts._instances():
    #    print i


    



