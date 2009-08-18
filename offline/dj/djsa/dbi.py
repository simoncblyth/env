

class DBI(dict):
    def __init__(self, dburl):
        """
            DB access and instrospection ... curtesy of SqlSoup 

            This object gets dict entries for each DBI table pair found :
                In [1]: dbi
                Out[1]: {u'SimPmtSpec': <class 'sqlalchemy.ext.sqlsoup.MappedJoin'>}

        """
        from sqlalchemy import create_engine, MetaData
        e = create_engine( dburl )
        from sqlalchemy.ext.sqlsoup import SqlSoup
        db = SqlSoup( MetaData(e) )
        self.db = db
        self.fk_joins()

    def fk_joins(self):
        """
            Manually apply FK joins between payload and validity DBI tables 
            This is required because MySQL(MyISAM) tables do not retain FK constraints
        """
        db = self.db
        for t in [n[0:-3] for n in db.engine.table_names() if n.endswith('Vld')]:
            self[t] = db.join( getattr(db,t), getattr(db,"%sVld"%t), getattr(db,t).SEQNO == getattr(db,"%sVld"%t).SEQNO , isouter=False )
 


if __name__=='__main__':
    from env.base.private import Private
    p = Private()
    dbi = DBI(p('DATABASE_URL'))

    print dbi['SimPmtSpec'].first()

