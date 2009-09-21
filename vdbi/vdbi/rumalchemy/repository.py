
from rum import repository

from rumalchemy import SARepositoryFactory, sqlsoup
from rumalchemy.util import get_mapper, get_foreign_keys

from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy.sql import expression
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import column_property
from sqlalchemy import join 

from inspect import isclass

import logging
log = logging.getLogger(__name__)

import os
from vdbi import PAY_COLUMNS, VLD_COLUMNS, JOIN_POSTFIX, VLD_POSTFIX
from vdbi.dbg import debug_here

from vdbi.rum.query import DbiQueryFactory








def fix_entity_name( entity , table ):
    entity.__name__ = str(table.name)    ## fix the .capitalized name  
    log.debug("fix_entity_name %s %s " % ( repr(entity), table ))

class DbiSARepositoryFactory(SARepositoryFactory):

    @repository.names_for_resource.when("isclass(resource)", prio=10)
    def _default_names_for_resource(self, resource):
        name = resource.__name__
        plural = "%ss" % name
        return name, plural       ## get template error if these are the same 
    
    @repository.names_for_resource.when("not isclass(resource)", prio=10)
    def _default_names_for_instance(self, resource):
        return self.names_for_resource(resource.__class__)
        
    def __call__(self, resource, parent=None, remote_name=None, action=None, parent_id=None):
        """
          Called 4 times for each table row :  3* for Dbi + 1 for Vld  
          due to RespositoryFactory.get_id from url_for : show/edit/delete dbi + related vld
            
          The original QueryFactory object that was hooked up in rum.Repository 
          is replaced by DbiQueryFactory 
        """
        get_repo = super(DbiSARepositoryFactory, self).__call__
        repo = get_repo(resource, parent, remote_name=remote_name, action=action)
        repo.queryfactory = DbiQueryFactory()     
        return repo
   
    def dbi_fk_ojoins(self, soup ):
        """
            Manually apply FK joins between payload and validity DBI tables 
            This is required because MySQL(MyISAM) tables do not retain FK constraints
            
            this is joining via the  object level ... hence the lowercased vld
            
            the rumalchemy sqlsoup does not have the join method
            
        """
        joins = []
        for t in [n[0:-3] for n in soup._metadata.tables.keys() if n.endswith(VLD_POSTFIX)]:
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
         pay_tables = [n[0:-3] for n,t in metadata.tables.items() if n.endswith(VLD_POSTFIX)]
         vld_tables = ["%s%s" % (n, VLD_POSTFIX) for n in pay_tables]
         for p,v in zip(pay_tables,vld_tables): 
             pay_t = metadata.tables.get(p, None )
             vld_t = metadata.tables.get(v, None )
             jo = join( pay_t , vld_t , pay_t.c.SEQNO == vld_t.c.SEQNO , isouter=False )
             jo.name = "%s%s" % ( p , JOIN_POSTFIX )
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
        pay_tables = [n[0:-3] for n,t in metadata.tables.items() if n.endswith(VLD_POSTFIX)]
        vld_tables = ["%s%s" % (n,VLD_POSTFIX) for n in pay_tables]    
        for p,v in zip(pay_tables,vld_tables): 
            pay = metadata.tables.get(p, None )
            vld = metadata.tables.get(v, None )
            if not(pay) or not(vld):
                print "skipping tables %s " % n
                continue
            pay.append_constraint( ForeignKeyConstraint( ['SEQNO'] , ['%s.SEQNO' % v ] ) )
        return vld_tables + pay_tables 
              
              
              
    def prepare_properties(self, table , related=False ):
        """
            using related succeeds to set column properties from 
            the related objects ... but they do not show up in the table
            or on the filter option list 
        """
        skips = PAY_COLUMNS.keys() + VLD_COLUMNS.keys()
        cols = [col for col in table.columns if col.name not in skips ] 
        prefix = os.path.commonprefix( [col.name for col in cols] )
        def attrname( col ):
            """
             change the mapped class attribute name for simpler a
             tighter presentation 
            """
            if PAY_COLUMNS.get(col.name,None):
                return PAY_COLUMNS.get(col.name)
            elif VLD_COLUMNS.get(col.name,None):
                return VLD_COLUMNS.get(col.name)
            elif len(prefix) == 0:
                return col.name 
            #return  "%s_%s" % ( prefix , col.name[len(prefix):] )
            return  col.name[len(prefix):] 
  
        from sqlalchemy.util import OrderedDict
        properties = OrderedDict()
        for col in table.columns:
            properties[attrname(col)] = col

        if related:
            if not(table.name.endswith(VLD_POSTFIX)):
                vt = self.soup._cache.get("%s%s" % (table.name,VLD_POSTFIX), None )
                assert vt
                print "paired vld mapped class vt : %s " % vt 
                for vc in vt._table.columns:
                    vn = "v_%s" % vc.name
                    properties[vn] = column_property( (vc).label(vn) )   
        return properties
              
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
                properties = self.prepare_properties(table) 
                kwargs = { 'properties':properties  }
                t = soup.class_for_table(table , **kwargs)
                fix_entity_name( t , table )     
            else:
                t = None
            soup._cache[attr] = t
        return t


    def _auto_relate(self, entities, mappers):
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
                        
                        e_name = self.names_for_resource(entity)[1]
                        p_name = self.names_for_resource(parent)[0]
                        ## get unicode error on attempting to print the bare entity 
                        log.debug( "_auto_relate table_name %s entity %s e_name %s p_name %s " % ( table_name, repr(entity), e_name , p_name ))
                        if entity._table.__class__.__name__ != 'Join':
                            assert getattr(self.soup,table_name) is entity
                        parent.relate( e_name, entity, backref=p_name , **relation_kwds)


    def _reflect_models(self):
        # Use the scoped_session sqlsoup creates. This is suboptimal, we
        # need a way to bring the objects sqlsoup creates into our
        # session (which we can control: set to transactional, etc...)
        # If we don't use soup's session SA barfs with  a
        # 'object is already attached to session blah'
        log.debug( "_reflect_models ... customized in %s " % self.__class__.__name__ )
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
        
 
        ## do not need relations from the joined tables ... so can do here  ?
        tjo = self.dbi_fk_tjoins( metadata )
        for tj in tjo:
            properties = self.prepare_properties(tj) 
            kwargs = { 'properties':properties  }
            entity = db.map( tj , **kwargs )
            fix_entity_name( entity , tj )
            entities[tj.name] = entity
            
        ## update the mappers 
        for e in entities.itervalues():
            if not(mappers.has_key(e)):
                mappers[e] = get_mapper(e)
 
        ## try relating the joined too 
        self._auto_relate( entities , mappers )
 
 
        self.mappers = mappers 
        self.entities = entities 
        
        ## set the resource names  ... causes peak.rules to complain of ambiguous methods for names_for_resources
        ##   http://docs.python-rum.org/developer/modules/genericfunctions.html
        ##for e in entities.itervalues():
        ##    self.set_names( e , e.__name__, e.__name__)
        
        return entities.values() 




def test_standalone_saquery():
    """
       standalone queries cause assertions due to lack of resource hookup 
       instead : access the query via the repo queryfactory (which is engine agnostic)
    """
    from rumalchemy.query import SAQuery
    sq =  SAQuery(and_([eq(u'SEQ', u'1'), eq(u'SITE', u'1'), eq(u'AD', u'1'), eq(u'RING', u'1')]), None, None, None)



def test_dump_columns():
    for modl in factory._models:
        mapr = factory.mappers[ modl ]
        print "modl %s mapr %s " % ( modl, mapr ) 
        print "modl.c %s " % list(modl.c)
        print "mapr column properties : "
        for cp in mapr.iterate_properties:
            print cp
    

def test_join_tables():
    metadata = factory.soup._metadata
    pay_t = metadata.tables.get("SimPmtSpec")
    vld_t = metadata.tables.get("SimPmtSpecVld")
    pv_t = join( pay_t , vld_t , pay_t.c.SEQNO == vld_t.c.SEQNO , isouter=False )
    
 
def test_select_all():
    """
        all[-1]  fails
    """
    all = repo.select()
    assert all[0].ROW == 2
    assert all[1000].ROW == 1002 
    assert all.count() == 3169
 

def test_select_ctx():
    """
        Makes use of the extended translate genericfunction that 
        understands ctx_ expressions 
    """
    q  = repo.queryfactory( resource ) 
    assert q.__class__.__name__ == 'SAQuery'
    
    sq = repo.select(q)      
    assert sq.__class__.__name__ == 'Query' and sq.__class__.__module__ == 'sqlalchemy.orm.query'
    
    from vdbi.rum.query import ctx_
    p  = q.clone( expr = ctx_({'Timestamp': u'2009/09/03 18:54', 'DetectorId': 0, 'SimFlag': 2, 'Site': 1})  )       
    sp = repo.select(p)
    
    assert sp[0].ROW == 2
    assert sp[1000].ROW == 1002
    assert sp.count() == 3169 
    
 


if __name__=='__main__':

    from sqlalchemy import create_engine
    from sqlalchemy.orm import scoped_session, sessionmaker
    from env.base.private import Private
    priv = Private()
    engine = create_engine(priv('DATABASE_URL'))    

    Session = scoped_session(sessionmaker(autocommit=False, autoflush=True))
    factory = DbiSARepositoryFactory( reflect='dbi', engine=engine, session_factory=Session)
    factory.load_resources()
    resource = factory._models[1]
    repo = factory(resource) 

    #test_dump_columns()
    #test_join_tables()
    test_select_all()
    test_select_ctx()


