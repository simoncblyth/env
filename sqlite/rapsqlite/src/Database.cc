#include "RapSqlite/Database.hh"
#include "RapSqlite/Table.hh"

const char* Database::SENTINEL = "COLUMNS" ; 
const int Database::DEBUG  = 0 ; 

const char* Database::Path(const char* envvar)
{
   const char* path = envvar ? getenv(envvar) : NULL ;
   return path ; 
}

Database::Database(const char* envvar) : m_debug(DEBUG), m_db(NULL), m_resultcolumn(0) 
{
   const char* path = Path(envvar);
   int rc = sqlite3_open(path, &m_db );
   if( rc ){
      fprintf(stderr, "Can't open database at path %s: %s\n", path, sqlite3_errmsg(m_db));
   }
   if(m_debug > 0) fprintf(stderr,"Opened %s \n", path);

   //Introspect();
}

Database::~Database()
{
   sqlite3_close(m_db);
}


int Database::LastInsertRowId()
{
    return sqlite3_last_insert_rowid(m_db);
}


void Database::SetDebug(int debug)
{
    m_debug = debug ;
}
int Database::GetDebug()
{
    return m_debug ;
}

void Database::ClearResults()
{
   m_results.clear();
   m_rows.clear();
}

std::vector<std::string>& Database::GetResults()
{
   return m_results; 
}
std::vector<std::string> Database::GetResultsCopy()
{
   return m_results; 
}


void Database::DumpResults(const char* msg)
{
   size_t size = m_results.size();
   printf("%s : size %zu \n", msg, size);
   for(size_t i=0 ; i<size ; ++i ) printf("%zu %s\n", i,m_results[i].c_str()); 
}

void Database::DumpRows(const char* msg)
{
   size_t size = m_rows.size();
   printf("%s : size %zu \n", msg, size);
   for(size_t i=0 ; i<size ; ++i ) DumpMap("row",m_rows[i]);
}

void Database::DumpMap(const char* msg, Map_t& map)
{
   printf("Database::DumpMap : %s \n", msg);
   for(Map_t::const_iterator it=map.begin() ; it != map.end() ; ++it ) printf(" %20s  : %s \n", it->first.c_str(), it->second.c_str() ); 
}


std::string Database::GetResult(int n)
{
   std::string empty ;
   return m_results.size() > n  ? m_results[n] : empty ;  
}


// pseudo-methodcall 
static int c_callback(void *self, int argc, char **argv, char **azColName)
{
    Database* db = reinterpret_cast<Database*>(self);
    return db->callback(argc, argv, azColName);
}

int Database::callback(int argc, char **argv, char **azColName)
{
   for(int i=0; i<argc; i++)
   {
      const char* key = azColName[i] ;
      const char* val = argv[i] ? argv[i] : "NULL" ; 
      // collecting only 1st column of results 
      if(i == 0) m_results.push_back(std::string(val));
      //printf("%s = %s ", key, val);
   }
   //printf("\n");
   return 0;
}


void Database::ExecCallback(const char* sql )
{
   printf("Database::ExecCallback [%s]\n", sql ); 
   char* zErrMsg = 0;
   ClearResults();
   int rc = sqlite3_exec(m_db, sql, c_callback, this, &zErrMsg);
   if( rc != SQLITE_OK )
   {
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
   }
}


char Database::Type(int type)
{
   char rc = '?';
   switch(type) 
   {
       case SQLITE_INTEGER:rc = 'i' ;break; 
       case SQLITE_FLOAT:rc = 'f' ;break; 
       case SQLITE_TEXT:rc = 't' ;break; 
       case SQLITE_BLOB:rc = 'b' ;break; 
       case SQLITE_NULL:rc = 'n' ;break; 
   }
   return rc ;
}


// exec and helpers


bool Database::Prepare(const char* sql, sqlite3_stmt** statement )
{
   if(m_debug>0) printf("Database::Prepare [%s]\n", sql ); 
   int rc = sqlite3_prepare_v2(m_db, sql, -1, statement, 0);
   if( rc != SQLITE_OK )
   {
       const char* err = sqlite3_errmsg(m_db);
       fprintf(stderr, "Database::Exec sqlite3_prepare_v2 error with sql %s : %s \n", sql, err );
   }
   return rc == SQLITE_OK ;
}

int Database::ExecI(const char* sql, int param)
{
   sqlite3_stmt* statement;
   bool ok = Prepare( sql, &statement ); 
   if(!ok) return -1 ;

   int rc = sqlite3_bind_int( statement, 1, param );
   if (rc != SQLITE_OK )
   {
       const char* err = sqlite3_errmsg(m_db);
       fprintf(stderr, "Database::ExecI sqlite3_bind_int error with sql %s : %s \n", sql, err );
       return -1 ; 
   }
   return Exec(sql, statement);
}

int Database::Exec(const char* sql)
{

   sqlite3_stmt* statement;
   bool ok = Prepare( sql, &statement ); 
   if(!ok) return -1 ;
   return Exec(sql, statement);
}


int Database::Exec(const char* sql, sqlite3_stmt* statement)
{
   int ncol = sqlite3_column_count(statement);
   if(m_debug > 0) printf("Database::Exec %s ncol %d \n", sql, ncol );

   m_results.clear();
   m_rows.clear();

   size_t count = 0 ;
   while(sqlite3_step(statement) == SQLITE_ROW )
   {
       if(count == 0) FillTypes(m_typelast, statement, ncol);
       Map_t row ; 
       FillColumns(row, statement, ncol );
       m_rows.push_back(row); 
       count++;
   }
   sqlite3_finalize(statement);
   if(m_debug>2) Dump();

   assert(m_rows.size() == count );
   return count ;
}

Map_t& Database::GetRowType()
{
   return m_typelast ;
}


Map_t Database::GetRow(std::size_t index, const char* sentinel)
{
    Map_t row ;
    if( index < m_rows.size() ) row = m_rows[index];
    if(sentinel != NULL ) row[std::string(sentinel)] = GetRowSpec();
    return row ; 
}

std::string Database::GetRowSpec()
{
   size_t size = m_typelast.size(); 
   Map_t::iterator it = m_typelast.begin();
   std::stringstream ss ; 
   for(size_t i = 0 ; i<size ; ++i )
   {
        ss << it->first << ":" << it->second ;
        if(i < size - 1) ss << "," ;
        it++ ;
   }
   return ss.str();
}

std::size_t Database::GetRowCount()
{
   return m_rows.size();
}
std::vector<Map_t>& Database::GetRows()
{
   return m_rows ;
}



void Database::Dump()
{
    DumpResults();
    DumpRows();
}


void Database::FillTypes(Map_t& typemap, sqlite3_stmt* statement, int ncol )
{
     typemap.clear();
     for(int c = 0; c < ncol; c++)
     {
         const char* name = sqlite3_column_name(statement, c);
         char type = Type(sqlite3_column_type(statement, c));
         typemap[std::string(name)] = std::string(&type, 1); 
     }
}

void Database::FillColumns(Map_t& rowmap, sqlite3_stmt* statement, int ncol )
{
     rowmap.clear();
     for(int c = 0; c < ncol; c++)
     {
         const char* name = sqlite3_column_name(statement, c);
         const char* text = (const char*)sqlite3_column_text(statement, c);
         rowmap[std::string(name)] = std::string(text);

         // for single column collection 
         if(c == m_resultcolumn) m_results.push_back(std::string(text));
     }
}




std::vector<long> Database::GetIVec(const char* column, const char* sql)
{   
    std::vector<long> ivec ;
    int rc = Exec(sql);
    if(rc < 0){
        printf("Database::GetIVec error from Exec %s %d \n", sql, rc );
        return ivec ;
    }

    std::string colname(column);
    for(int r = 0; r < m_rows.size(); r++)
    {
        Map_t row = m_rows[r];

        {
            const char* val = row[colname].c_str();
            char* end;
            long ival = strtol(val,&end,10);
            if(end == val)
            {
                printf("Database::GetIVec failed to extract int from column %s string %s \n", column, val );
            }
           ivec.push_back(ival);  
        }

    }
    return ivec ; 
}









void Database::SetResultColumn(std::size_t col)
{
   m_resultcolumn = col ; 
}

std::size_t Database::GetResultColumn()
{
   return m_resultcolumn ; 
}


void Database::AddTable(Table* table)
{
   std::string tn(table->GetName());
   Table* prior = FindTable(tn.c_str());
   if(prior==NULL)
   {
       m_tables[tn] = table ;
   }
   else
   {
       fprintf(stderr,"WARNING: replacing table \n"); // leak 
       m_tables[tn] = table ;
   }
}

Table* Database::FindTable(const char* name)
{
   std::string tn(name);
   return m_tables.find(tn) == m_tables.end() ? NULL : m_tables[tn] ;
}


void Database::IntrospectTableNames()
{
    std::string listall = Table::ListAllStatement();
    SetResultColumn(0);
    this->Exec(listall.c_str());
    m_tablenames = GetResultsCopy();
    //DumpTableNames();
}

void Database::DumpTableNames()
{
    for(int i=0 ; i<m_tablenames.size(); ++i ) fprintf(stderr, "%s\n", m_tablenames[i].c_str());
}

void Database::IntrospectTableSQL()
{
    SetResultColumn(0);
    for(int i=0 ; i<m_tablenames.size(); ++i )
    {
        const char* tn = m_tablenames[i].c_str() ;
        std::string tablesql = Table::TableSQLStatement(tn);
        this->Exec(tablesql.c_str());

        std::string sql = GetResult(0);
        printf("Database::IntrospectTableSQL %d name %s sql %s \n", i, tn, sql.c_str()  );
    }
}

void Database::IntrospectTableInfo()
{
    for(int i=0 ; i<m_tablenames.size(); ++i )
    {
        const char* tn = m_tablenames[i].c_str() ;
        std::string tableinfo = Table::TableInfoStatement(tn);
        //printf("Database::IntrospectTableInfo %d name %s \n", i, tn );

        std::vector<std::string> names ; 
        std::vector<std::string> types ; 

        SetResultColumn(1);
        this->Exec(tableinfo.c_str());
        names = m_results ; 

        SetResultColumn(2);
        this->Exec(tableinfo.c_str());
        types = m_results ; 

        SetResultColumn(0);

        assert(names.size() == types.size());

        Table* table = new Table(tn);
        for(int c=0 ; c < names.size() ; ++c )
        {
            table->AddColumn( names[c].c_str(), types[c].c_str() ); 
        }
        this->AddTable(table); 
    }
}


void Database::DumpTables()
{
   for(TableMap_t::iterator it=m_tables.begin() ; it != m_tables.end() ; it++)
   {
        Table* table = it->second ; 
        table->Dump();
   }
}


void Database::Introspect()
{
    IntrospectTableNames();
    //IntrospectTableSQL();
    IntrospectTableInfo();
}


void Database::Create(const char* tn, const char* spec )
{
    Map_t map = dsplit(spec, ',', ':');
    Create(tn, map); 
}
void Database::Insert(const char* tn, const char* spec )
{
    Map_t map = dsplit(spec, ',', ':');
    Insert(tn, map); 
}
void Database::Create(const char* tn, Map_t& map, const char* columns)
{
    Table* t = new Table(tn);
    t->AddDefinition(map, columns);  // when non-NULL columns controls ordering  
    this->AddTable(t);

    Table* chk = this->FindTable(tn);
    assert( chk == t );
    
    std::string create = t->CreateStatement();
    this->Exec(create.c_str()); 
}
void Database::Insert(const char* table, Map_t& map, const char* columns)
{
    Table* t = this->FindTable(table);
    if(!t) return ; 

    std::string insert = t->InsertStatement(map);
    int rc = this->Exec(insert.c_str());
    if(rc != 0)
    {
        printf("Database::Insert into table %s failed \n", table );
        for(Map_t::iterator it=map.begin() ; it!=map.end() ; it++)
        {
            printf(" %20s : %s \n", it->first.c_str(), it->second.c_str()) ;
        } 
    }
}


void Database::Select(const char* table)
{
    Table* t = this->FindTable(table);
    if(!t) return ; 
    std::string select = t->SelectStatement();
    this->Exec(select.c_str());
}


