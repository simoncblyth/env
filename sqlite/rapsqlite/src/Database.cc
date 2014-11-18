#include "RapSqlite/Database.hh"
#include "RapSqlite/Table.hh"


void Database::ClearResults()
{
   m_results.clear();
}

std::vector<std::string>& Database::GetResults()
{
   return m_results; 
}
std::vector<std::string> Database::GetResultsCopy()
{
   return m_results; 
}


void Database::DumpResults()
{
   size_t size = m_results.size();
   for(size_t i=0 ; i<size ; ++i )
   {
       printf("%zu %s\n", i,m_results[i].c_str()); 
   }
}




std::string Database::GetResult(int n)
{
   std::string empty ;
   return m_results.size() > n  ? m_results[n] : empty ;  
}




Database::Database(const char* envvar) : m_db(NULL), m_resultcolumn(0) 
{
   const char* path = getenv(envvar);
   int rc = sqlite3_open(path, &m_db );
   if( rc ){
      fprintf(stderr, "Can't open database at path %s: %s\n", path, sqlite3_errmsg(m_db));
   }
   //fprintf(stderr,"Opened %s \n", path);
}


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


void Database::Exec(const char* sql, int debug )
{
   ClearResults();
   if(debug>0) printf("Database::Exec [%s]\n", sql ); 

   sqlite3_stmt *statement;

   int rc = sqlite3_prepare_v2(m_db, sql, -1, &statement, 0);
   if( rc != SQLITE_OK )
   {
       const char* err = sqlite3_errmsg(m_db);
       fprintf(stderr, "Database::Exec sqlite3_prepare_v2 error with sql %s : %s \n", sql, err );
       return ;
   }

   int ncol = sqlite3_column_count(statement);
   char* types = new char[ncol+1];
   types[0] = '\0';

   int first = 1 ; 
   while(sqlite3_step(statement) == SQLITE_ROW )
   {
       if(first) 
       {
           for(int c = 0; c < ncol; c++)
           {
               types[c] = Type(sqlite3_column_type(statement, c));
               const char* decl = sqlite3_column_decltype(statement, c);
               const char* name = sqlite3_column_name(statement, c);
               if(debug>1) printf(" %s:[%c]%s ", name, types[c],decl ); 
           }
           if(debug>1) printf("\n");
           types[ncol] = '\0' ;
           first = 0 ;
       }

       if(debug>2) printf("%s ", types);
       for(int c = 0; c < ncol; c++)
       {
           const char* text = (const char*)sqlite3_column_text(statement, c);
           if(c == m_resultcolumn) m_results.push_back(std::string(text));
           if(debug>2) printf(" %s ", text );
       }
       if(debug>2) printf("\n");
   }
   sqlite3_finalize(statement);

   if(debug>0) DumpResults();
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
    this->Exec(listall.c_str(),0);
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
        this->Exec(tablesql.c_str(),0);

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
        this->Exec(tableinfo.c_str(),0);
        names = m_results ; 

        SetResultColumn(2);
        this->Exec(tableinfo.c_str(),0);
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
        //std::string tn = it->first ;
        Table* table = it->second ; 
        //printf(" tn %s \n", tn.c_str() );
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
void Database::Create(const char* tn, Map_t& map )
{
    Table* t = new Table(tn);
    t->AddDefinition(map);
    this->AddTable(t);

    Table* chk = this->FindTable(tn);
    assert( chk == t );
    
    std::string create = t->CreateStatement();
    this->Exec(create.c_str()); 
}
void Database::Insert(const char* table, Map_t& map)
{
    Table* t = this->FindTable(table);
    if(!t) return ; 

    std::string insert = t->InsertStatement(map);
    this->Exec(insert.c_str());
}
void Database::Select(const char* table)
{
    Table* t = this->FindTable(table);
    if(!t) return ; 
    std::string select = t->SelectStatement();
    this->Exec(select.c_str());
}
Database::~Database()
{
   sqlite3_close(m_db);
}


