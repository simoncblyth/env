

from rumalchemy import SARepositoryFactory, sqlsoup
from rumalchemy.util import get_mapper, get_foreign_keys

from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy.sql import expression
from sqlalchemy import MetaData, Table

import os

SKIP_COLUMNS = ('SEQNO','ROW_COUNTER')


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
              
    def entity(self, soup, attr ):
        """ pull this out of the soup to allow better control of the mapping """
        try:
            t = soup._cache[attr]
        except KeyError:
            table = Table(attr, soup._metadata, autoload=True, schema=soup.schema)

            ## initially tried trimming ... but this messes up the ordering ... so insert an undercore to allow the interface to split 
            if not table.primary_key.columns:
                raise PKNotFoundError('table %r does not have a primary key defined [columns: %s]' % (attr, ','.join(table.c.keys())))
            if table.columns:
                
                cols = [col for col in table.columns if col.name not in SKIP_COLUMNS] 
                prefix = os.path.commonprefix( [col.name for col in cols] )
                def attrname( col ):
                    if col.name in SKIP_COLUMNS or len(prefix) == 0:return col.name 
                    return  "%s_%s" % ( prefix , col.name[len(prefix):] )
                #properties = {}
                #wargs = { 'properties':properties , 'include_properties':[attrname(col) for col in table.columns]   }
                #print kwargs
                t = soup.class_for_table(table )
                 
                # avoid setting the properties via the dict, as this looses the column ordering 
                mapr = get_mapper( t )
                for col in cols:
                    mapr.add_property( attrname(col) , col )

                


            else:
                t = None
            soup._cache[attr] = t
        return t


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
                #entities[table_name]=db.entity(table_name)
                entities[table_name]=self.entity(db, table_name )
            except sqlsoup.PKNotFoundError:
                log.warn("reflection: skipping table "+table_name+ "...")

        ## the soup messes up the column ordering ... fix it
        #for e in entities.itervalues():
        #    e.c = expression.ColumnCollection()
        #    t = e._table
        #    mapr = get_mapper(e)
        #    props = {}
        #    for k in mapr.iterate_properties:
        #        props[k.columns[0].name] = k
        #    for col in t.columns:
        #        k = props.get( col.name , None )
        #        if k:
        #            e.c[k.key] = k.columns[0]
        # 


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



if __name__=='__main__':

    from sqlalchemy import create_engine
    from sqlalchemy.orm import scoped_session, sessionmaker
    from env.base.private import Private
    p = Private()
    engine = create_engine(p('DATABASE_URL'))    

    Session = scoped_session(sessionmaker(autocommit=False, autoflush=True))
    factory = DbiSARepositoryFactory( reflect='dbi', engine=engine, session_factory=Session)
    factory.load_resources()

    for modl in factory._models:
        mapr = factory.mappers[ modl ]
        print modl, mapr 
        print list(modl.c)
        for cp in mapr.iterate_properties:
            print cp



