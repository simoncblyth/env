#ifndef RSDATABASE_H
#define RSDATABASE_H

#include "RapSqlite/Common.hh"

class Database {
public:
   static const int DEBUG ;
   static const char* SENTINEL ;

   static const char* Path(const char* envvar );
   Database(const char* envvar );
   virtual ~Database();

   // TODO: introspect table definitions from DB schema queries
   void AddTable(Table* table);
   Table* FindTable(const char* name);

   int callback(int argc, char **argv, char **azColName);

   void Introspect();
   void IntrospectTableNames();
   void IntrospectTableSQL();
   void IntrospectTableInfo();

public:
   // Exec and helpers, result collection 
   int Exec(const char* sql);
   int ExecI(const char* sql, int param);
   int Exec(const char* sql, sqlite3_stmt* statement);

   bool Prepare(const char* sql, sqlite3_stmt** statement );
   void FillColumns(Map_t& rowmap, sqlite3_stmt* statement, int ncol );
   void FillTypes(Map_t& typemap, sqlite3_stmt* statement, int ncol );

public:
   // single column result handling 
   void SetResultColumn(std::size_t rc); 
   std::size_t GetResultColumn(); 
   std::vector<std::string>& GetResults();
   std::vector<std::string>  GetResultsCopy();
   std::string GetResult(int n=0);

public:
   // map result handling
   void DumpMap(const char* msg, Map_t& map);
   std::size_t GetRowCount();
   std::vector<Map_t>& GetRows();
   Map_t GetRow(std::size_t index=0, const char* sentinel=SENTINEL);
   Map_t& GetRowType();
   std::string GetRowSpec();
   std::vector<long> GetIVec(const char* column, const char* sql);

   void SetDebug(int debug);
   int GetDebug();
   void DumpTableNames();
   void DumpTables();
   void Select(const char* table);
   void Create(const char* table, Map_t& map, const char* columns=NULL);
   void Insert(const char* table, Map_t& map, const char* columns=NULL);
   void Create(const char* tn, const char* spec );
   void Insert(const char* tn, const char* spec );
   int LastInsertRowId();

   void ClearResults();

   void DumpResults(const char* msg="Database::DumpResults");
   void DumpRows(const char* msg="Database::DumpRows");
   void Dump();

   char Type(int type);
   void ExecCallback(const char* sql );

private:
   int m_debug ;
   sqlite3* m_db;
   std::vector<std::string> m_tablenames ; 
   std::map<std::string,Table*> m_tables ; 
   std::vector<std::string> m_results ; 
   std::size_t m_resultcolumn ; 
   std::vector<Map_t> m_rows ; 
   Map_t m_typelast ; 

};

#endif

