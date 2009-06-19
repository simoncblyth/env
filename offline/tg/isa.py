

class DBI(dict):
    def __init__(self, dburl):
        """
            DB access and instrospection ...
        """
        from sqlalchemy import create_engine, MetaData
        e = create_engine( dburl )
        from sqlalchemy.ext.sqlsoup import SqlSoup
        db = SqlSoup( MetaData(e) )
        self.db = db

    def __call__(self):
        """
            Manually apply FK joins between payload and validity DBI tables 
            This is required because MySQL(MyISAM) tables do not retain FK constraints
        """
        db = self.db
        for t in [n[0:-3] for n in self.db.engine.table_names() if n.endswith('Vld')]:
            self[t] = db.join( getattr(db,t), getattr(db,"%sVld"%t), getattr(db,t).SEQNO == getattr(db,"%sVld"%t).SEQNO , isouter=False )
        return self



if __name__=='__main__':
    from env.base.private import Private
    p = Private()
    dbi = DBI(p('DATABASE_URL'))()

    print dbi['SimPmtSpec'].first()

