#ifndef RSTABLE_H
#define RSTABLE_H

#include "RapSqlite/Common.hh"

class Table {
public:
   static const char* INTEGER_TYPE;
   static const char* FLOAT_TYPE;
   static const char* STRING_TYPE;
   static const char* BLOB_TYPE;

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


#endif
