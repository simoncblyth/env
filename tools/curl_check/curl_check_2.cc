/**
curl_check_2.cc
=================

CHECK=2 ~/e/tools/curl_check/curl_check.sh


**/


#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <cassert>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curl/curl.h>



struct NP_CURL_HDR
{
    NP_CURL_HDR(const char* name);

    std::string name ; 

    std::string level ;
    std::string dtype ;
    std::string shape ;

    std::string c_length ;
    std::string c_type ; 

    void collect( const char* name, const char* value );
    std::string desc() const ;

    static constexpr const char* x_numpy_level = "x-numpy-level" ; // debug level integer
    static constexpr const char* x_numpy_dtype = "x-numpy-dtype" ;
    static constexpr const char* x_numpy_shape = "x-numpy-shape" ;
    static constexpr const char* content_length = "content-length" ; 
    static constexpr const char* content_type   = "content-type" ; 

    static  std::string Format(const char* prefix, const char* value);
    static  std::string Format_LEVEL(const char* level);
    static  std::string Format_DTYPE(const char* dtype);
    static  std::string Format_SHAPE(const char* shape);

    static bool Expected_DTYPE(const char* dtype);
}; 

inline NP_CURL_HDR::NP_CURL_HDR( const char* name_ )
    :
    name(name_)
{
}


inline void NP_CURL_HDR::collect( const char* name, const char* value )
{
    if(      0==strcmp(name,x_numpy_level)) level = value ;
    else if( 0==strcmp(name,x_numpy_dtype)) dtype = value ;
    else if( 0==strcmp(name,x_numpy_shape)) shape = value ;
    else if( 0==strcmp(name,content_length)) c_length = value ;
    else if( 0==strcmp(name,content_type))   c_type = value ;
}

inline std::string NP_CURL_HDR::desc() const
{
    std::stringstream ss ; 
    ss << "[NP_CURL_HDR::desc [" << name << "]\n" ;
    ss << std::setw(20) << x_numpy_level << " : " << level << "\n" ;
    ss << std::setw(20) << x_numpy_dtype << " : " << dtype << "\n" ;
    ss << std::setw(20) << x_numpy_shape << " : " << shape << "\n" ;
    ss << std::setw(20) << content_length << " : " << c_length << "\n" ;
    ss << std::setw(20) << content_type   << " : " << c_type << "\n" ;

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


inline std::string NP_CURL_HDR::Format_LEVEL( const char* level ){ return Format(x_numpy_level, level ); }
inline std::string NP_CURL_HDR::Format_DTYPE( const char* dtype ){ return Format(x_numpy_dtype, dtype ); }
inline std::string NP_CURL_HDR::Format_SHAPE( const char* shape ){ return Format(x_numpy_shape, shape ); }

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





struct Upload 
{
    const char *data;
    size_t size;
    static size_t read_callback(void *buffer, size_t size, size_t nitems, void *userdata) ;
};

inline size_t Upload::read_callback(void *buffer, size_t size, size_t nitems, void *userdata) 
{
    struct Upload* upload = (struct Upload *)userdata;
    size_t copy_size = size * nitems;

    if (copy_size > upload->size) copy_size = upload->size; // for buffered read make sure to stay in range

    memcpy(buffer, upload->data, copy_size);

    upload->data += copy_size;  // move data pointer
    upload->size -= copy_size;  // decrease remaining size

    return copy_size;
}



struct Download {
    char *buffer;
    size_t size;
    std::string desc() const ;

    static size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) ;

};

inline std::string Download::desc() const
{
    std::stringstream ss ; 
    ss << "Download::desc " << size << "\n" ;
    std::string str = ss.str() ;
    return str ; 
}

inline size_t Download::write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) 
{
    struct Download* download = (struct Download *)userdata;
    size_t new_len = download->size + (size * nmemb);
    char *new_buffer = (char*)realloc(download->buffer, new_len + 1);

    if (new_buffer == NULL) {
        // Realloc failed, a real-world app would handle this more robustly
        fprintf(stderr, "realloc() failed!\n");
        return 0; // Abort transfer
    }

    // Update the buffer pointer and size
    download->buffer = new_buffer;
    memcpy(&(download->buffer[download->size]), ptr, size * nmemb);
    download->buffer[new_len] = '\0';
    download->size = new_len;

    return size * nmemb;
}





int main(void) 
{


    const char* url = "http://127.0.0.1:8000/upload_array" ;
    std::string level = "1" ;

    std::vector<float> a = { 0.f, 1.f, 2.f, 3.f };


    std::string dtype = "float32" ;

    const char* upload_payload = (const char*)a.data() ;
    size_t upload_size = a.size()*sizeof(float); 
    std::string shape = std::to_string(a.size()); 


    std::cout << " upload_size " << upload_size << "\n" ;



    struct Upload upload = {
        .data = upload_payload,
        .size = upload_size
    };

    struct Download download = {
        .buffer = (char*)malloc(1), // Start with a 1-byte buffer
        .size = 0
    };
    download.buffer[0] = '\0';



    NP_CURL_HDR down("down") ;



    CURL* curl;
    CURLcode result ;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) 
    {
        bool expected_dtype = NP_CURL_HDR::Expected_DTYPE(dtype.c_str());
        assert( expected_dtype );

        std::string x_level = NP_CURL_HDR::Format_LEVEL(level.c_str()) ;
        std::string x_dtype = NP_CURL_HDR::Format_DTYPE(dtype.c_str()) ;
        std::string x_shape = NP_CURL_HDR::Format_SHAPE(shape.c_str()) ;

        struct curl_slist* headerlist = nullptr ;
        headerlist = curl_slist_append(headerlist, x_level.c_str() );
        headerlist = curl_slist_append(headerlist, x_dtype.c_str() );
        headerlist = curl_slist_append(headerlist, x_shape.c_str() );

        curl_easy_setopt(curl, CURLOPT_URL, url );
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);

        curl_easy_setopt(curl, CURLOPT_READFUNCTION, Upload::read_callback);
        curl_easy_setopt(curl, CURLOPT_READDATA, &upload );
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)upload.size);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, Download::write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &download );

        result = curl_easy_perform(curl);

        if (result != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(result));

        struct curl_header *h;
        struct curl_header *prev = NULL;
        do {
            h = curl_easy_nextheader(curl, CURLH_HEADER, -1, prev);
            if(h) down.collect(h->name, h->value);
            prev = h;
        } while(h);
 

        curl_easy_cleanup(curl);
        curl_slist_free_all(headerlist);

        std::cout << download.desc() ;

        free(download.buffer);
    }


    std::cout << down.desc() ; 


    curl_global_cleanup();
    return 0;
}



