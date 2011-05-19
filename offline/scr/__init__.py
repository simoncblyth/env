"""
Generic Scraping Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~

Goals
  * eliminate duplication in scraper code
  * reduce scraper code to managable levels, that do not scale with the tables scraped
  
GOAL OF SA MAPPING IS NOT A WHOLE-IN-ONE, as need to use DybDbi on the destination side anyhow
  * source mapped instances are an intermediary 
  * ensemble intermediary needed ... for delta logic

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
 
Note that scraper code table names do not match current offline_db : DcsPmtHv

"""

if __name__ == '__main__':
    pass







