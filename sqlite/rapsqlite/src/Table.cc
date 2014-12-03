#include "RapSqlite/Table.hh"


const char* Table::INTEGER_TYPE = "integer" ;
const char* Table::FLOAT_TYPE = "real" ;
const char* Table::STRING_TYPE = "text" ;
const char* Table::BLOB_TYPE = "blob" ;
const char* Table::PK = "id integer primary key" ;

    
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
       printf(" %3zu %10s : %s \n", i, m_type[i].c_str(), m_keys[i].c_str() );
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


Table* Table::FromCreateStatement(const char* sql)
{
   //
   // attempting to parse sql create statement : obsoleted by use of "pragma table_info(tablename)"
   //
    assert(0); // this is obsolete

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


std::string Table::CreateStatement(const char* pk)
{
    assert( m_keys.size() == m_type.size() ); 
    std::stringstream ss ;
    ss << "create table if not exists " << m_name << " (" ; 

    if(pk) ss << pk << "," ;

    for(size_t i=0 ; i < m_keys.size() ; ++i )
    {
        ss << m_keys[i] << " " << m_type[i] ;
        if( i < m_keys.size() - 1 ) ss << "," ; 
    }
    ss << ");" ;
    return ss.str();  
}

std::string Table::InsertStatement(Map_t& map, const char* pk)
{
    // TODO: binding approaches ? so can do this once and reuse with different binds

    std::stringstream ss ;
    ss << "insert into " << m_name << " values(" ;
    if(pk) ss << "null," ; // auto incrementing PK

    for(size_t i=0 ; i < m_keys.size() ; ++i )
    {
       if(map.find(m_keys[i]) != map.end())
       {
           std::string val = map[m_keys[i]] ;
           if(m_type[i] == STRING_TYPE )
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



