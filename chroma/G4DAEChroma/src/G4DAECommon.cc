#include "G4DAEChroma/G4DAECommon.hh"

#include <sstream>
#include <cassert>
#include "G4AffineTransform.hh"

#include "md5digest.h"
#include <time.h>   

#ifdef WITH_ZMQ
#include <zmq.h>
#endif


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

   for (int i = 0; i < buflen ; i++){
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
   for (int i = 0; i < v.size() ; i++){
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





void current_time(char* buf, int buflen, const char* tfmt, int utc)
{
   time_t t;
   time (&t); 
   struct tm* tt = utc ? gmtime(&t) : localtime(&t) ;
   strftime(buf, buflen, tfmt, tt);
}



std::string now(const char* tfmt, const int buflen, int utc )
{
    char buf[buflen];
    current_time( buf, buflen, tfmt, utc );  
    return std::string(buf);
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

    rc = zmq_msg_recv (&msg, socket, 0);   

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


