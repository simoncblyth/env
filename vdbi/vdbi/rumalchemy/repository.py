

from rumalchemy import SARepositoryFactory, sqlsoup
from rumalchemy.util import get_mapper, get_foreign_keys

from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy import MetaData


class DbiSARepositoryFactory(SARepositoryFactory):
   
    def dbi_fkc( self, metadata ):
        """
           Append FK constraints to the SA tables as MySQL dont know em 

        """
        from sqlalchemy import ForeignKeyConstraint
        pay_tables = [n[0:-3] for n,t in metadata.tables.items() if n.endswith('Vld')]
        vld_tables = ["%sVld" % n for n in pay_tables]    
        for p,v in zip(pay_tables,vld_tables): 
            pay = metadata.tables.get(p, None )
            vld = metadata.tables.get(v, None )
            if not(pay) or not(vld):
                print "skipping tables %s " % n
                continue
            pay.append_constraint( ForeignKeyConstraint( ['SEQNO'] , ['%s.SEQNO' % v ] ) )
        return pay_tables + vld_tables
              
    def _reflect_models(self):
        # Use the scoped_session sqlsoup creates. This is suboptimal, we
        # need a way to bring the objects sqlsoup creates into our
        # session (which we can control: set to transactional, etc...)
        # If we don't use soup's session SA barfs with  a
        # 'object is already attached to session blah'
        print "_reflect_models ... customized in %s " % self.__class__
        metadata = MetaData(self.engine)
        metadata.reflect()
   
        dbi_tables = self.dbi_fkc( metadata )     
         
        # workaround: sqlalchemy does unicode names reflecting and SqlSoup
        #             doesn't like it
        for name, table in metadata.tables.iteritems():
            table.name = str(name)
            metadata.tables[table.name] = metadata.tables.pop(name)
        
        if self.reflect == 'all':
            table_names = metadata.tables.keys()
        elif self.reflect == 'dbi':
            table_names = dbi_tables 
        else:
            table_names = self.reflect
        
        db = sqlsoup.SqlSoup(metadata,self.session_factory)

        entities=dict()
        for table_name in table_names:
            try:
                entities[table_name]=db.entity(table_name)
            except sqlsoup.PKNotFoundError:
                log.warn("reflection: skipping table "+table_name+ "...")
        mappers = dict((e, get_mapper(e)) for e in entities.itervalues())
        # autogenerate relations
        for table_name, entity in entities.iteritems():
            self._fix_soup_entity(entity)
            for prop in mappers[entity].iterate_properties:
                if isinstance(prop, ColumnProperty):
                    for col in prop.columns:
                        # See if the column is a foreign key
                        try:
                            fk = get_foreign_keys(col)[0]
                        except IndexError:
                            # It isn't...
                            continue
                        # It is, lookup parent mapper
                        relation_kwds=dict()
                        for parent, m in mappers.iteritems():
                            if fk.references(m.local_table):
                                if col.primary_key:
                                    relation_kwds["cascade"]='all, delete-orphan'
                                break
                        # Relate it

                        assert getattr(db,table_name) is entity
# 
                        parent.relate(
                            self.names_for_resource(entity)[1],entity,
                            backref=self.names_for_resource(parent)[0],
                            **relation_kwds)
        self.soup = db  
        self.mappers = mappers 
        self.entities = entities 
        
        return entities.values()




