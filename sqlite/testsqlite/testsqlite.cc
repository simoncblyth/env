/*
   see testsqlite-

   clang testsqlite.cc -L/opt/local/lib -lsqlite3 -lstdc++  -o $LOCAL_BASE/env/bin/testsqlite 

   DBPATH=/tmp/testsqlite.db testsqlite

*/

#include <stdio.h>
#include <cstdlib>
#include <sqlite3.h> 

#include <cassert>
#include <sstream>
#include <string>
#include <vector>
#include <map>


class Table ; 

typedef std::vector<std::string> Vec_t ; 
typedef std::map<std::string,std::string> Map_t ; 
typedef std::map<std::string,Table*> TableMap_t ; 


class Table {
public:

   static std::string ListAllStatement();
   static std::string TableSQLStatement(const char* name);
   static std::string TableInfoStatement(const char* name);
   static Table* FromCreateStatement(const char* sql);

   Table(const char* name);
   const char* GetName();

   void AddColumn( const char* key, const char* type );
   void AddDefinition(Map_t& map);
   void Dump();

   std::size_t GetNumColumns();

   virtual ~Table();

   std::string CreateStatement();
   std::string SelectStatement();
   std::string InsertStatement(Map_t& map);

private:
   std::string m_name ;
   Vec_t m_keys ; 
   Vec_t m_type ; 
};



    
Table::Table(const char* name) : m_name(name) {}
Table::~Table(){}

std::size_t Table::GetNumColumns()
{
    assert(m_keys.size() == m_type.size());
    return m_keys.size();
}


const char* Table::GetName()
{
    return m_name.c_str();
}

void Table::AddDefinition(Map_t& map)
{
    for(Map_t::iterator it=map.begin() ; it!=map.end(); it++ )
    {
       this->AddColumn( it->first.c_str(), it->second.c_str() ); 
    }   
}

void Table::Dump()
{
    printf("Table::Dump %s \n", m_name.c_str());
    size_t ncol = GetNumColumns();
    for(size_t i=0 ; i < ncol ; ++i ){
       printf(" %zu   %s %s \n", i, m_keys[i].c_str(), m_type[i].c_str() );
    }
}



std::string Table::ListAllStatement()
{
    std::stringstream ss ;
    ss << "select tbl_name from sqlite_master where type=\"table\" ;" ; 
    return ss.str();
}   
std::string Table::TableInfoStatement(const char* name)
{
    std::stringstream ss ;
    ss << "pragma table_info(" << name << ");" ; 
    return ss.str();
}
std::string Table::TableSQLStatement(const char* name)
{
    std::stringstream ss ;
    ss << "select sql from sqlite_master where type=\"table\" and tbl_name = \"" << name << "\" ;"  ;
    return ss.str();
}


void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ;
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}



Table* Table::FromCreateStatement(const char* sql)
{
    std::string create(sql);
    size_t bop = create.find('(')+1; 
    size_t bcl = create.find(')');  // hmm eg char(50) type messes this up.. need to find last 
    std::string cols = create.substr(bop, bcl-bop);

    std::vector<std::string> elem ;
    split(elem, cols.c_str(), ',' );

    printf(" bop %zu bcl %zu cols: %s  #col %lu \n", bop, bcl, cols.c_str(), elem.size() );

    for(int i=0 ; i<elem.size() ; ++i)
    {
       printf(" %d %s \n", i, elem[i].c_str() );
    } 


    return NULL ;
}


std::string Table::CreateStatement()
{
    assert( m_keys.size() == m_type.size() ); 
    std::stringstream ss ;
    ss << "create table if not exists " << m_name << " (" ; 
    for(size_t i=0 ; i < m_keys.size() ; ++i )
    {
        ss << m_keys[i] << " " << m_type[i] ;
        if( i < m_keys.size() - 1 ) ss << "," ; 
    }
    ss << ");" ;
    return ss.str();  
}

std::string Table::InsertStatement(Map_t& map)
{
    std::stringstream ss ;
    ss << "insert into " << m_name << " values(" ;
    for(size_t i=0 ; i < m_keys.size() ; ++i )
    {
       if(map.find(m_keys[i]) != map.end())
       {
           std::string val = map[m_keys[i]] ;
           if(m_type[i] == "string" )
           {
               ss << "\"" << val << "\"" ; 
           }
           else
           {
               ss << val ; 
           } 
       }
       else
       {
           ss << "null"  ;
       } 
       if( i < m_keys.size() - 1 ) ss << "," ; 
    }
    ss << ");" ;
    return ss.str();  
}

std::string Table::SelectStatement()
{
    std::stringstream ss ;
    ss << "select * from " << m_name << " ;" ;
    return ss.str();  
}

void Table::AddColumn(const char* key, const char* type)
{
    m_keys.push_back(std::string(key));
    m_type.push_back(std::string(type));
}





class DB {
public:
   DB(const char* envvar );

   // TODO: introspect table definitions from DB schema queries
   void AddTable(Table* table);
   Table* FindTable(const char* name);

   int callback(int argc, char **argv, char **azColName);

   void Introspect();
   void IntrospectTableNames();
   void IntrospectTableSQL();
   void IntrospectTableInfo();

   void DumpTableNames();
   void DumpTables();
   void Select(const char* table);
   void Create(const char* table, Map_t& map);
   void Insert(const char* table, Map_t& map);


   void SetResultColumn(std::size_t rc); 
   std::size_t GetResultColumn(); 

   void ClearResults();
   void DumpResults();
   std::vector<std::string>& GetResults();
   std::vector<std::string>  GetResultsCopy();

   std::string GetResult(int n=0);

   void Exec(const char* sql, int debug=0 );
   char Type(int type);
   void ExecCallback(const char* sql );
   virtual ~DB();

private:
   std::size_t m_resultcolumn ; 
   std::vector<std::string> m_tablenames ; 
   std::map<std::string,Table*> m_tables ; 
   std::vector<std::string> m_results ; 

   sqlite3* m_db;
};



void DB::ClearResults()
{
   m_results.clear();
}

std::vector<std::string>& DB::GetResults()
{
   return m_results; 
}
std::vector<std::string> DB::GetResultsCopy()
{
   return m_results; 
}


void DB::DumpResults()
{
   size_t size = m_results.size();
   for(size_t i=0 ; i<size ; ++i )
   {
       printf("%zu %s\n", i,m_results[i].c_str()); 
   }
}




std::string DB::GetResult(int n)
{
   std::string empty ;
   return m_results.size() > n  ? m_results[n] : empty ;  
}




DB::DB(const char* envvar) : m_db(NULL), m_resultcolumn(0) 
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
    DB* db = reinterpret_cast<DB*>(self);
    return db->callback(argc, argv, azColName);
}

int DB::callback(int argc, char **argv, char **azColName)
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


void DB::ExecCallback(const char* sql )
{
   printf("DB::ExecCallback [%s]\n", sql ); 
   char* zErrMsg = 0;
   ClearResults();
   int rc = sqlite3_exec(m_db, sql, c_callback, this, &zErrMsg);
   if( rc != SQLITE_OK )
   {
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
   }
}


char DB::Type(int type)
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


void DB::Exec(const char* sql, int debug )
{
   ClearResults();
   if(debug>0) printf("DB::Exec [%s]\n", sql ); 

   sqlite3_stmt *statement;

   int rc = sqlite3_prepare_v2(m_db, sql, -1, &statement, 0);
   if( rc != SQLITE_OK )
   {
       const char* err = sqlite3_errmsg(m_db);
       fprintf(stderr, "DB::Exec sqlite3_prepare_v2 error with sql %s : %s \n", sql, err );
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


void DB::SetResultColumn(std::size_t col)
{
   m_resultcolumn = col ; 
}

std::size_t DB::GetResultColumn()
{
   return m_resultcolumn ; 
}


void DB::AddTable(Table* table)
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

Table* DB::FindTable(const char* name)
{
   std::string tn(name);
   return m_tables.find(tn) == m_tables.end() ? NULL : m_tables[tn] ;
}


void DB::IntrospectTableNames()
{
    std::string listall = Table::ListAllStatement();
    SetResultColumn(0);
    this->Exec(listall.c_str(),0);
    m_tablenames = GetResultsCopy();
    //DumpTableNames();
}

void DB::DumpTableNames()
{
    for(int i=0 ; i<m_tablenames.size(); ++i ) fprintf(stderr, "%s\n", m_tablenames[i].c_str());
}

void DB::IntrospectTableSQL()
{
    SetResultColumn(0);
    for(int i=0 ; i<m_tablenames.size(); ++i )
    {
        const char* tn = m_tablenames[i].c_str() ;
        std::string tablesql = Table::TableSQLStatement(tn);
        this->Exec(tablesql.c_str(),0);

        std::string sql = GetResult(0);
        printf("DB::IntrospectTableSQL %d name %s sql %s \n", i, tn, sql.c_str()  );
    }
}

void DB::IntrospectTableInfo()
{
    for(int i=0 ; i<m_tablenames.size(); ++i )
    {
        const char* tn = m_tablenames[i].c_str() ;
        std::string tableinfo = Table::TableInfoStatement(tn);
        //printf("DB::IntrospectTableInfo %d name %s \n", i, tn );

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


void DB::DumpTables()
{
   for(TableMap_t::iterator it=m_tables.begin() ; it != m_tables.end() ; it++)
   {
        //std::string tn = it->first ;
        Table* table = it->second ; 
        //printf(" tn %s \n", tn.c_str() );
        table->Dump();
   }
}


void DB::Introspect()
{
    IntrospectTableNames();
    //IntrospectTableSQL();
    IntrospectTableInfo();
}

void DB::Create(const char* tn, Map_t& map )
{
    Table* t = new Table(tn);
    t->AddDefinition(map);
    this->AddTable(t);

    Table* chk = this->FindTable(tn);
    assert( chk == t );
    
    std::string create = t->CreateStatement();
    this->Exec(create.c_str()); 
}
void DB::Insert(const char* table, Map_t& map)
{
    Table* t = this->FindTable(table);
    if(!t) return ; 

    std::string insert = t->InsertStatement(map);
    this->Exec(insert.c_str());
}
void DB::Select(const char* table)
{
    Table* t = this->FindTable(table);
    if(!t) return ; 
    std::string select = t->SelectStatement();
    this->Exec(select.c_str());
}
DB::~DB()
{
   sqlite3_close(m_db);
}



int main()
{
    DB* db = new DB("DBPATH");

    db->Introspect();
    //db->DumpTables();

    Map_t map ; 
    map["x"] = "int" ;
    map["y"] = "string" ;
    map["z"] = "float" ;
    map["w"] = "float" ;
    db->Create("C", map );


    map["x"] = "1" ;
    map["y"] = "hello" ;
    map["z"] = "1.1" ;
    map["w"] = "2.2" ;
    db->Insert("C",map);


    db->Select("C");

    delete db ;  
}


