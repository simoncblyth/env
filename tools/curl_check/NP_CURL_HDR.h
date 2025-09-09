#pragma once

struct NP_CURL_HDR
{
    NP_CURL_HDR(const char* name);

    std::string name ; 

    std::string token ;
    int         level ;
    std::string dtype ;
    std::string shape ;
    std::vector<int> sh ; 

    size_t      c_length ;
    std::string c_type ; 
    std::string content ; // usually empty, only populated after HTTP error with json response

    void collect( const char* name, const char* value );
    void collect_json_content( char* buffer, size_t size );

    std::string sstr() const ;
    std::string desc() const ;

    static constexpr const char* x_numpy_token = "x-numpy-token" ;
    static constexpr const char* x_numpy_level = "x-numpy-level" ; // debug level integer
    static constexpr const char* x_numpy_dtype = "x-numpy-dtype" ;
    static constexpr const char* x_numpy_shape = "x-numpy-shape" ;

    static constexpr const char* content_length = "content-length" ; 
    static constexpr const char* content_type   = "content-type" ; 

    static  std::string Format(const char* prefix, const char* value);
    static  std::string Format_TOKEN(const char* token);
    static  std::string Format_LEVEL(const char* level);
    static  std::string Format_DTYPE(const char* dtype);
    static  std::string Format_SHAPE(const char* shape);

    static void Parse_SHAPE( std::vector<int>& sh, const char* shape );


    static bool Expected_DTYPE(const char* dtype);
}; 

inline NP_CURL_HDR::NP_CURL_HDR( const char* name_ )
    :
    name(name_)
{
}

inline void NP_CURL_HDR::collect( const char* name, const char* value )
{
    if( 0==strcmp(name,x_numpy_level))
    {
        std::stringstream ss(value);
        ss >> level ; 
    }
    else if( 0==strcmp(name,x_numpy_token)) 
    {
        token = value ;
    }
    else if( 0==strcmp(name,x_numpy_dtype)) 
    {
        dtype = value ;
    }
    else if( 0==strcmp(name,x_numpy_shape)) 
    {
        shape = value ;
        Parse_SHAPE(sh, shape.c_str());
    }
    else if( 0==strcmp(name,content_length))
    {
        std::stringstream ss(value);
        ss >> c_length ; 
    }
    else if( 0==strcmp(name,content_type))
    {
         c_type = value ;
    }
}

inline void NP_CURL_HDR::collect_json_content( char* buffer, size_t size )
{
    if( c_type == "application/json" && c_length > 0 && c_length == size )
    {
        content.resize( size + 1  );  // +1 ?
        memcpy(content.data(), buffer, size ) ;
        content.data()[size] = '\0' ; 
    }
}


inline std::string NP_CURL_HDR::sstr() const 
{
    int num = sh.size();
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++) ss << sh[i] << ( i < num - 1 ? "," : " " ) ;
    std::string str = ss.str() ; 
    return str ; 
}


inline std::string NP_CURL_HDR::desc() const
{
    std::stringstream ss ; 
    ss << "[NP_CURL_HDR::desc [" << name << "]\n" ;
    ss << std::setw(20) << x_numpy_token << " : " << token << "\n" ;
    ss << std::setw(20) << x_numpy_level << " : " << level << "\n" ;
    ss << std::setw(20) << x_numpy_dtype << " : " << dtype << "\n" ;
    ss << std::setw(20) << x_numpy_shape << " : " << shape << "\n" ;
    ss << std::setw(20) << "sh.size"     << " : " << sh.size() << "\n" ;
    ss << std::setw(20) << "sstr"        << " : " << sstr() << "\n" ;
    ss << std::setw(20) << content_length << " : " << c_length << "\n" ;
    ss << std::setw(20) << content_type   << " : " << c_type << "\n" ;
    ss << std::setw(20) << "json_content" << "\n" << content << "\n" ;

    ss << "]NP_CURL_HDR::desc\n" ;
    std::string str = ss.str();
    return str ;
}


inline std::string NP_CURL_HDR::Format( const char* prefix, const char* value )
{
    std::stringstream ss ; 
    ss << prefix << ":" << value ; 
    std::string str = ss.str();
    return str ;
} 

inline std::string NP_CURL_HDR::Format_TOKEN( const char* token ){ return Format(x_numpy_token, token ); }
inline std::string NP_CURL_HDR::Format_LEVEL( const char* level ){ return Format(x_numpy_level, level ); }
inline std::string NP_CURL_HDR::Format_DTYPE( const char* dtype ){ return Format(x_numpy_dtype, dtype ); }
inline std::string NP_CURL_HDR::Format_SHAPE( const char* shape ){ return Format(x_numpy_shape, shape ); }

inline void NP_CURL_HDR::Parse_SHAPE( std::vector<int>& sh, const char* shape )
{
    int num;
    std::stringstream ss;
    for (int i=0 ; i < int(strlen(shape)) ; i++) ss << (std::isdigit(shape[i]) ? shape[i] : ' ' ) ; // replace non-digits with spaces  
    while (ss >> num) sh.push_back(num); 
}


inline bool NP_CURL_HDR::Expected_DTYPE(const char* dtype)
{
     return    strcmp(dtype, "float32") == 0 
            || strcmp(dtype, "float64") == 0 
            || strcmp(dtype, "int64")   == 0 
            || strcmp(dtype, "int32")   == 0 
            || strcmp(dtype, "int16")   == 0 
            || strcmp(dtype, "int8")    == 0 
            || strcmp(dtype, "uint64")  == 0 
            || strcmp(dtype, "uint32")  == 0 
            || strcmp(dtype, "uint16")  == 0 
            || strcmp(dtype, "uint8")   == 0 
               ;
}




