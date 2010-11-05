from sqlalchemy import create_engine, MetaData 
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.sqlsoup import SqlSoup
from datetime import datetime

class DbiSoup(dict):
    def __init__(self, dburl):
        """
            DB access and instrospection ... curtesy of sqlalchemy + SqlSoup extension

            Usage : 
                from private import Private
                p = Private()
                dbis = DBISoup( p('DATABASE_URL') )
                locals().update(dbis)    ## populates scope with the mapped classes 

        """
        self.engine = create_engine( dburl )
        self.insp = Inspector.from_engine(self.engine)
        self.meta = MetaData(self.engine) 
        self.soup = SqlSoup( self.meta )

        self.pk_check()
        self.map_all()

    all_tables = property( lambda self:self.insp.get_table_names() ) 
    vld_tables = property( lambda self:filter(lambda t:t[-3:].upper() == "VLD", self.all_tables ))
    pay_tables = property( lambda self:map(lambda t:t[0:-3], self.vld_tables ))
    dbi_pairs  = property( lambda self:zip(self.pay_tables, self.vld_tables))

    def map_all(self):
        for p,v in self.dbi_pairs:
            self(p)
                
    def __call__(self, t ):
        """
            accessing the attribute from the soup, pulls the class into existance  
        """
        p,v = (t, "%sVld" % t)
        pay = getattr( self.soup , p )
        vld = getattr( self.soup , v )
        self[p] = self.soup.join( pay, vld, pay.SEQNO == vld.SEQNO , isouter=False )  ## pay + vld join 
        self[v] = vld 
        return self[p]
 
    def pk_check(self):
        """
            Inspector API is new in SA 0.6.5?
        """
        for p,v in self.dbi_pairs:
            for t in p,v:
                pks = self.insp.get_primary_keys(t)
                cols = self.insp.get_columns(t)
                assert len(cols) > 1 , cols
                if t == p:
                    assert cols[0]['name'] == 'SEQNO' and cols[1]['name'] == 'ROW_COUNTER'
                    assert pks == ['SEQNO','ROW_COUNTER']
                elif t == v:
                    assert cols[0]['name'] == 'SEQNO'
                    assert pks == ['SEQNO']


if __name__=='__main__':
    from private import Private
    p = Private()
    dbis = DbiSoup( p('DATABASE_URL') )
    locals().update(dbis)

    assert SimPmtSpec.count() == 2546 
    assert CalibPmtSpec.count() == 4160 
    assert SimPmtSpec.get((1,100)).VERSIONDATE == datetime(2010, 1, 20, 0, 0)   ## CPK get 
    assert CalibFeeSpecVld.count() == 111


