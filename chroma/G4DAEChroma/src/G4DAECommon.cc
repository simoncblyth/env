#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/md5digest.h"

#include <sstream>
#include <cassert>
#include "G4AffineTransform.hh"


#ifdef WITH_ZMQ
#include <zmq.h>
#endif


#include <unistd.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>


using namespace std ; 


void DumpBuffer(const char* buffer, size_t buflen, size_t maxlines ) 
{
   const char* hfmt = "  %s \n%06X : " ;

   int ascii[2] = { 0x20 , 0x7E };
   const int N = 16 ;
   size_t halfmaxbytes = N*maxlines/2 ; 

   char line[N+1] ;
   int n = N ;
   line[n] = '\0' ;
   while(n--) line[n] = ' ' ;

   for (size_t i = 0; i < buflen ; i++){
       int v = buffer[i] & 0xff ;
       bool out = i < halfmaxbytes || i > buflen - halfmaxbytes - 1 ; 
       if( i == halfmaxbytes || i == buflen - halfmaxbytes - 1  ) printf(hfmt, "...", i );  
       if(!out) continue ; 

       int j = i % N ;
       if(j == 0) printf(hfmt, line, i );  // output the prior line and start new one with byte counter  
       line[j] = ( v >= ascii[0] && v < ascii[1] ) ? v : '.' ;  // ascii rep 
       printf("%02X ", v );
   }
   printf(hfmt, line, buflen );
   printf("\n"); 
}


void DumpVector(const std::vector<float>& v, size_t itemsize) 
{
   const char* hfmt = "\n%04d : " ;
   for (size_t i = 0; i < v.size() ; i++){
       if(i % itemsize == 0) printf(hfmt, i ); 
       printf("%10.3f ", v[i]);
   }
   printf(hfmt, v.size() );
   printf("\n"); 
}






string md5digest( const char* buffer, int len )
{
    char* out = md5digest_str2md5(buffer, len);
    string digest(out); 
    free(out);
    return digest;
}

string transform_rep( G4AffineTransform& transform )
{

   G4RotationMatrix rotation = transform.NetRotation();
   G4ThreeVector rowX = rotation.rowX();
   G4ThreeVector rowY = rotation.rowY();
   G4ThreeVector rowZ = rotation.rowZ();
   G4ThreeVector tran = transform.NetTranslation(); 
   
   stringstream ss; 
   ss << tran << " " << rowX << rowY << rowZ  ;
   return ss.str();
}



void split( vector<string>& elem, const char* line, char delim )
{
    if(line == NULL){ 
        cout << "split NULL line not defined : " << endl ; 
        return ;
    }   
    istringstream f(line);
    string s;
    while (getline(f, s, delim)) elem.push_back(s);
}

void isplit( vector<int>& elem, const char* line, char delim )
{
    if(line == NULL){ 
        cout << "isplit NULL line not defined : " << endl ; 
        return ;
    }   
    istringstream f(line);
    string s;
    while (getline(f, s, delim)) elem.push_back(atoi(s.c_str()));
}


void getintpair( const char* range, char delim, int* a, int* b ) 
{
    if(!range) return ;

    std::vector<std::string> elem ;   
    split(elem, range, delim);
    assert( elem.size() == 2 );

    *a = atoi(elem[0].c_str()) ;
    *b = atoi(elem[1].c_str()) ;
}

void getinttriplet( const char* range, char delim, int* a, int* b, int* c ) 
{
    if(!range) return ;

    std::vector<std::string> elem ;   
    split(elem, range, delim);
    assert( elem.size() == 3 );

    *a = atoi(elem[0].c_str()) ;
    *b = atoi(elem[1].c_str()) ;
    *c = atoi(elem[2].c_str()) ;
}






// return path up to the last occurrence of the delim
// in principal the path returned should be free-d
char* basepath( const char* _path, char delim )
{
    char* path = strdup(_path);
    char* dot  = strrchr(path, delim) ;  // returns NULL when delim not found
    if(dot) *dot = '\0' ;
    return path ; 
}


std::string join(std::vector<std::string>& elem, char delim )
{
    typedef std::vector<std::string> Vec_t ;
    std::stringstream ss ;    
    for(size_t i=0 ; i < elem.size() ; ++i)
    {
        ss << elem[i] ;
        if( i < elem.size() - 1) ss << delim ; 
    }
    return ss.str();
}


std::string removeField(const char* line, char delim, int index )
{
    //  
    //   split the line with the delim
    //   then reassemble skipping the field pointed to by rfield
    //  
    //    For example the below line with delim '.' and rfield -2
    //  
    //       /path/to/geometry.dae.noextra.abcdefghijklmnopqrstuvwxyz.dae
    //       /path/to/geometry.dae.noextra.dae
    //  

    std::vector<std::string> elem ;
    split(elem, line, delim);      

    if(index >= 0 && index < elem.size())
    {
        elem.erase( elem.begin() + index);
    }
    else if( index < 0 && -index < elem.size())
    {
        elem.erase( elem.end() + index );
    } 
    else
    {
        printf("removeField line %s delim %c index %d : invalid index \n", line, delim, index );
    }
    return join(elem, delim); 
}


std::string insertField(const char* line, char delim, int index, const char* field)
{
    std::vector<std::string> elem ;
    split(elem, line, delim);      

    std::string s(field);

    if(index >= 0 && index < elem.size())
    {
        elem.insert( elem.begin() + index, s);
    }
    else if( index < 0 && -index < elem.size())
    {
        elem.insert( elem.end() + index, s );
    } 
    else
    {
        printf("insertField line %s delim %c index %d : invalid index \n", line, delim, index );
    }
    return join(elem, delim); 
}







int mkdirp(const char* _path, int mode) 
{
    // directory tree creation by swapping slashes for end of string '\0'
    // then restoring the slash 
    //
    // NB when given a file path to be created this does NOT do the
    // the right thing : it creates a directory named like intended filepath 
    //  
    //  http://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
    //  printf("_path %s \n", _path);

    char* path = strdup(_path);
    char* p = path ;
    int rc = 0 ; 

    while (*p != '\0') 
    {   
        p++;
        while(*p != '\0' && *p != '/') p++;

        char v = *p;  // hold on to the '/'
        *p = '\0';
            
        //printf("path [%s] \n", path);

        rc = mkdir(path, mode);

        if(rc != 0 && errno != EEXIST) 
        {   
            *p = v;
            rc = 1;
            break ;
        }   
        *p = v;
    }   

    free(path); 
    return rc; 
}













#ifdef WITH_ZMQ
// Receive 0MQ string from socket and convert into C string



char* s_recv (void *socket) 
{
    zmq_msg_t message;
    zmq_msg_init (&message);
    int size = zmq_msg_recv (&message, socket, 0); 
    if (size == -1) return NULL;
    char* str  = (char*)malloc(size + 1);
    memcpy (str, zmq_msg_data (&message), size); zmq_msg_close (&message);
    str [size] = 0;
    return (str);
}


// Convert C string to 0MQ string and send to socket



int s_send (void *socket, char *str) 
{
    zmq_msg_t message;
    zmq_msg_init_size (&message, strlen(str));
    memcpy (zmq_msg_data (&message), str, strlen(str)); 
    int size = zmq_msg_send (&message, socket, 0); 
    zmq_msg_close (&message);
    return (size);
}



int b_send( void* socket, const char* bytes, size_t size, int flags )
{
   zmq_msg_t zmsg;
   int rc = zmq_msg_init_size (&zmsg, size);
   assert (rc == 0);
   
   memcpy(zmq_msg_data (&zmsg), bytes, size );   // TODO : check for zero copy approaches

   rc = zmq_msg_send (&zmsg, socket, flags);

   if (rc == -1) {
       int err = zmq_errno();
       printf ("b_send : Error occurred during zmq_msg_send : %s\n", zmq_strerror(err));
   }
   zmq_msg_close (&zmsg); 

#ifdef VERBOSE
   int nbytes = rc ; 
   printf ("b_send : zmq_msg_send sent %d bytes \n", nbytes);
#endif

   return rc ;
}



int b_recv( void* socket, zmq_msg_t& msg )
{

    int rc = zmq_msg_init (&msg); 
    assert (rc == 0);


    //

    rc = zmq_msg_recv (&msg, socket, 0);   


    //
    // simple method above exits with "Interrupted system call" 
    // on resizing terminal windows
    // this frailty can be avoided using while loop and sleeping 
    // at performance cost from the sleep
    //
    // TODO: work out a better way, maybe with a poller to avoid the 
    //       problem without having to sleep 
    //
    // http://stackoverflow.com/questions/16212526/how-can-i-clean-up-properly-when-recv-is-blocking
    //
    //

/*   
    while(-1 == zmq_msg_recv(&msg, socket, ZMQ_DONTWAIT))
    {
        if (EAGAIN != errno){
             break ;
        }  
        printf("b_recv sleeping\n");
        sleep(1);
    }
*/

    if(rc == -1){
        int err = zmq_errno();
        printf( "b_recv : Error on zmq_msg_recv : %s \n", zmq_strerror(err)) ;
        return rc ;
    } 

#ifdef VERBOSE
    printf( "b_recv : zmq_msg_recv received %d bytes \n", rc ) ;
#endif
    return rc ;
}


#endif







