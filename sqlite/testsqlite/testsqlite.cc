/*
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

typedef std::vector<std::string> Vec_t ; 
typedef std::map<std::string,std::string> Map_t ; 


class Table {
public:

   static std::string ListAllStatement();
   static std::string IntrospectStatement(const char* name);
   static Table* FromCreateStatement(const char* sql);

   Table(const char* name);
   const char* GetName();

   void AddColumn( const char* key, const char* type );
   void AddDefinition(Map_t& map);
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


std::string Table::ListAllStatement()
{
    std::stringstream ss ;
    ss << "select tbl_name from sqlite_master where type=\"table\" ;" ; 
    return ss.str();
}   

std::string Table::IntrospectStatement(const char* name)
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
   void Select(const char* table);
   void Create(const char* table, Map_t& map);
   void Insert(const char* table, Map_t& map);

   void ClearResults();
   std::vector<std::string>& GetResults();
   std::vector<std::string>  GetResultsCopy();

   std::string GetResult(int n=0);

   void Exec(const char* sql );
   virtual ~DB();

private:
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

std::string DB::GetResult(int n)
{
   std::string empty ;
   return m_results.size() > n  ? m_results[n] : empty ;  
}




DB::DB(const char* envvar) : m_db(NULL)
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


void DB::Exec(const char* sql )
{
   printf("DB::Exec [%s]\n", sql ); 
   char* zErrMsg = 0;
   ClearResults();
   int rc = sqlite3_exec(m_db, sql, c_callback, this, &zErrMsg);
   if( rc != SQLITE_OK )
   {
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
   }
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


void DB::Introspect()
{
    std::string listall = Table::ListAllStatement();
    this->Exec(listall.c_str()); 

    Vec_t tables = GetResultsCopy();

    for(int i=0 ; i<tables.size(); ++i )
    {
        const char* tn = tables[i].c_str() ;
        std::string introspect = Table::IntrospectStatement(tn);
        this->Exec(introspect.c_str());

        std::string sql = GetResult(0);
        printf("DB::Introspect table %d name %s sql %s \n", i, tn, sql.c_str()  );

        Table* t = Table::FromCreateStatement( sql.c_str());


    }
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

    Map_t map ; 
    map["x"] = "int" ;
    map["y"] = "string" ;
    map["z"] = "float" ;
    db->Create("B", map );



    map["x"] = "1" ;
    map["y"] = "hello" ;
    map["z"] = "1.1" ;
    db->Insert("B",map);

    map["x"] = "2" ;
    map["y"] = "world" ;
    map["z"] = "101.1" ;
    db->Insert("B",map);


    db->Select("B");
    delete db ;  
}


