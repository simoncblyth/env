"""
Generic Scraping Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~

Goals
  * eliminate duplication in scraper code
  * reduce scraper code to managable levels, that do not scale with the tables scraped
  
GOAL OF SA MAPPING IS NOT A WHOLE-IN-ONE, as need to use DybDbi on the destination side anyhow
  * source mapped instances are an intermediary 
  * ensemble intermediary needed ... for delta logic

MAYBE EASIER NOT TO USE SA MAPPING   

Boundary conditions
  * 2 databases : source and destination
  * sa mapping of source tables (or any selectable eg with join), to instances  
  * propagate such instances into DybDbi rows   
  * ``.spec``ify destination DBI table pairs  
     *  ? mapping specifics in the .spec ? 

dcs tables have nasty habit of encoding content (stuff that should be in rows) into table and column names 
this requires very non-standard mapping ... one row of source dcs table corresponds to many rows 
of destination table (actually ... need to map a subset ) 

  * http://www.sqlalchemy.org/docs/07/orm/mapper_config.html#sql-expressions-as-mapped-attributes
      combining columns into attributes
  * http://sqlalchemy.readthedocs.org/en/latest/orm/mapper_config.html#mapping-a-subset-of-table-columns  

   mapper(User, users_table, include_properties=['user_id', 'user_name'])

  * google:"sqlalchemy mapping multiple instances from single row"
  * http://stackoverflow.com/questions/1300433/how-to-map-one-class-against-multiple-tables-with-sqlalchemy


Are mapping a row to a collection of instances 
  * http://www.sqlalchemy.org/docs/orm/collections.html
  * http://stackoverflow.com/questions/780774/sqlalchemy-dictionary-of-tags
 
Note that scraper code is not matching current offline_db table

mysql> describe DcsPmtHv  ;
+-------------+--------------+------+-----+---------+----------------+
| Field       | Type         | Null | Key | Default | Extra          |
+-------------+--------------+------+-----+---------+----------------+
| SEQNO       | int(11)      | NO   | PRI |         |                | 
| ROW_COUNTER | int(11)      | NO   | PRI | NULL    | auto_increment | 
| ladder      | tinyint(4)   | YES  |     | NULL    |                | 
| col         | tinyint(4)   | YES  |     | NULL    |                | 
| ring        | tinyint(4)   | YES  |     | NULL    |                | 
| voltage     | decimal(6,2) | YES  |     | NULL    |                | 
| pw          | tinyint(4)   | YES  |     | NULL    |                | 
+-------------+--------------+------+-----+---------+----------------+
7 rows in set (0.14 sec)


"""



class PmtHv(object):
    """
    Table specifics that cannot be factored in the general scraper  

    ... hmm add a meta key in the DybDbi table spec to point at this class
    """
    dbi_table = "DcsPmtHv"  
    time_interval = 60
    dcstime = "date_time"

    def __init__(self, site, detector ):
        self.site = site
        self.detector = detector

    def selectable(self, db ):
        return join( db.tables[
  

class DcsScraper(object):
    """
    Generic aspects of scraping tables from DCS database are contained here
    """
    def __init__(self, kls ):
        self.kls = kls
        self.off = sa_meta(sa_url("copy_offline_db"))



if __name__ == '__main__':
    kls = PmtHv()
    scr = DcsScraper( kls )







