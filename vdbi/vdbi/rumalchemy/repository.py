

from rumalchemy import SARepositoryFactory, sqlsoup
from rumalchemy.util import get_mapper, get_foreign_keys

from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy.sql import expression
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import column_property
from sqlalchemy import join 

import os

SKIP_COLUMNS = ('SEQNO','ROW_COUNTER')


class DbiSARepositoryFactory(SARepositoryFactory):
   
   
    def dbi_fk_ojoins(self, soup ):
        """
            Manually apply FK joins between payload and validity DBI tables 
            This is required because MySQL(MyISAM) tables do not retain FK constraints
            
            this is joining via the  object level ... hence the lowercased vld
            
            the rumalchemy sqlsoup does not have the join method
            
        """
        joins = []
        for t in [n[0:-3] for n in soup._metadata.tables.keys() if n.endswith('Vld')]:
            payn = t
            vldn = payn + "Vld"
            payo = self.entity( soup, payn )
            vldo = self.entity( soup, vldn )
            print "joining %s and %s " % ( repr(payo) , repr(vldo) )
            ## failing with a unicode error
            jo = join( payo , vldo , payo.SEQ == vldo.SEQ , isouter=False )
            mjo = soup.map( jo )
            joins.append( mjo )
        return joins

    def dbi_fk_tjoins(self, metadata ):
         """
              Try to join at the table level, then map to the join ?
              
         """
         tjo = []
         pay_tables = [n[0:-3] for n,t in metadata.tables.items() if n.endswith('Vld')]
         vld_tables = ["%sVld" % n for n in pay_tables]
         for p,v in zip(pay_tables,vld_tables): 
             pay_t = metadata.tables.get(p, None )
             vld_t = metadata.tables.get(v, None )
             jo = join( pay_t , vld_t , pay_t.c.SEQNO == vld_t.c.SEQNO , isouter=False )
             jo.name = "%s_" % p
             tjo.append( jo )
         return tjo
   
    def dbi_fk_constraint( self, metadata ):
        """
           Append FK constraints to the SA tables as MySQL dont know em 
           and returns the names of the paired DBI tables found
           
           NB the vld table names are returned first, in order to be able to use
           them when mapping the payload tables 
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
        return vld_tables + pay_tables 
              
    def entity(self, soup, attr ):
        """ 
             This is pulled out of the soup to allow better control of the mapping 
             any common prefix on column names is removed for 
             the mapped class attribute names for a tighter table
             
             Had to pull this out of the soup as need access to the table columns
             prior to doing the mapping in order to prepare the attribute properties
             that effect the renaming. 
    
        """
        try:
            t = soup._cache[attr]
        except KeyError:
            table = Table(attr, soup._metadata, autoload=True, schema=soup.schema)

            if not table.primary_key.columns:
                raise PKNotFoundError('table %r does not have a primary key defined [columns: %s]' % (attr, ','.join(table.c.keys())))
            if table.columns:
                
                cols = [col for col in table.columns if col.name not in SKIP_COLUMNS] 
                prefix = os.path.commonprefix( [col.name for col in cols] )
                def attrname( col ):
                    """
                       change the mapped class attribute name for simpler a
                       tighter presentation 
                    """
                    if col.name in SKIP_COLUMNS:
                        return col.name[0:3]
                    elif len(prefix) == 0:
                        return col.name 
                    #return  "%s_%s" % ( prefix , col.name[len(prefix):] )
                    return  col.name[len(prefix):] 
                    
                from sqlalchemy.util import OrderedDict
                properties = OrderedDict()
                for col in table.columns:
                    properties[attrname(col)] = col
                
                if not(table.name.endswith("Vld")):
                    vt = soup._cache.get("%sVld" % table.name, None )
                    assert vt
                    print "paired vld mapped class vt : %s " % vt 
                    for vc in vt._table.columns:
                        vn = "v_%s" % vc.name
                        properties[vn] = column_property( (vc).label(vn) )
                
                kwargs = { 'properties':properties  }
                t = soup.class_for_table(table , **kwargs)

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
        print "_reflect_models ... customized in %s " % self.__class__.__name__
        metadata = MetaData(self.engine)
        metadata.reflect()
        self.metadata = metadata
   
        dbi_tables = self.dbi_fk_constraint( metadata )     
         
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
        self.soup = db  

        entities=dict()
        for table_name in table_names:
            try:
                ## pulling thru the entity maps the classes into existance 
                #entities[table_name]=db.entity(table_name)
                entities[table_name]=self.entity(db, table_name )
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
 
 
        ## do not need relations from the joined tables ... so can do here  ?
        tjo = self.dbi_fk_tjoins( metadata )
        for tj in tjo:
            entities[tj.name] = db.map( tj )

        ## update the mappers 
        for e in entities.itervalues():
            if not(mappers.has_key(e)):
                mappers[e] = get_mapper(e)
 
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
        print "modl %s mapr %s " % ( modl, mapr ) 
        print "modl.c %s " % list(modl.c)
        print "mapr column properties : "
        for cp in mapr.iterate_properties:
            print cp

    
    metadata = factory.soup._metadata
    pay_t = metadata.tables.get("SimPmtSpec")
    vld_t = metadata.tables.get("SimPmtSpecVld")
    pv_t = join( pay_t , vld_t , pay_t.c.SEQNO == vld_t.c.SEQNO , isouter=False )


