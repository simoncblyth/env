from sqlalchemy.engine.reflection import Inspector

class DbiSoup(dict):
    def __init__(self, soup):
        """
            DB access and instrospection ... curtesy of sqlalchemy + SqlSoup extension
                http://www.sqlalchemy.org/docs/orm/extensions/sqlsoup.html

                http://www.sqlalchemy.org/docs/orm/extensions/sqlsoup.html#sessions-transations-and-application-integration
                     the soup saves a lot pf effort, but unclear regarding integration 

            Multi-DB usage ...
               http://www.sqlalchemy.org/docs/05/session.html#vertical-partitioning
                   done by configuring the Session class

                http://www.sqlalchemy.org/docs/05/reference/orm/sessions.html#sqlalchemy.orm.sessionmaker

        """
        self.soup = soup
        self.insp = Inspector.from_engine(soup.engine)

        self.pk_check()
        self.map_all()

    all_tables = property( lambda self:self.insp.get_table_names() ) 
    vld_tables = property( lambda self:filter(lambda t:t[-3:].upper() == "VLD", self.all_tables ))
    pay_tables = property( lambda self:map(lambda t:t[0:-3], self.vld_tables ))
    dbi_pairs  = property( lambda self:zip(self.pay_tables, self.vld_tables))
    dbi_tables = property( lambda self:self.pay_tables + self.vld_tables )
    oth_tables = property( lambda self:filter(lambda t:t not in self.dbi_tables,self.all_tables))

    def map_all(self):
        for p,v in self.dbi_pairs:
            self.pairing(p,v)
        for t  in self.oth_tables:
            self[t] = getattr( self.soup , t )
                
    def pairing(self, p, v  ):
        """
            accessing the attribute from the soup, pulls the class into existance  
        """
        pay = getattr( self.soup , p )
        vld = getattr( self.soup , v )
        self[p] = self.soup.join( pay, vld, pay.SEQNO == vld.SEQNO , isouter=False )  ## pay + vld join 
        self[v] = vld 
        return self[p]
 

    def pk_check(self):
        """
             THIS BELONGS ELSEWHERE

            Inspector API is new in SA 0.6.5?
        """
        for p,v in self.dbi_pairs:
            for t in p,v:
                print t 
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
    pass



