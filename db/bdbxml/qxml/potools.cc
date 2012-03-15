#include "potools.hh"

/*
Config files such as:: 

	[container.path]
	path = /tmp/hfagc/hfagc.dbxml
	path = /tmp/hfagc/hfagc_system.dbxml

	[container.tag]
	tag = hfc
	tag = sys

Are converted into map with keys ("hfc","sys") and values ("/tmp/hfagc/hfagc.dbxml",..)

*/
void kv_zip( po::variables_map& vm , ssmap& ss , string kname , string vname )
{
      if (vm.count(kname) && vm.count(vname))
      {
           svec keys = vm[kname].as<svec>();
           svec vals  = vm[vname].as<svec>();
           size_t nkeys = keys.size() ; 
           if( nkeys == vals.size() ){
	       for(size_t i = 0 ; i < nkeys ; ++i )
               {
	            //cout << kname << " " << keys[i] << " " << vname << " " << vals[i] <<  endl ;
		    ss[keys[i]] = vals[i] ;
               }
           } else {
	       cout << "ERROR : numbers of " << kname << " and  " << vname << " must match " << endl ;
           } 		   
      }
}


/*
For single dotted keys such as dbxml.name from config such as::  

	  [dbxml]
	  name = a
	  other = b

Fill in the map with keys ("dbxml.name", "dbxml.other") and values ("a","b")

*/
void kv_pluck( po::variables_map& vm , ssmap& ss , int ndot )
{
      po::variables_map::const_iterator it ;
      for( it = vm.begin() ; it != vm.end() ; ++it ){
         string key(it->first) ; 
         size_t ff = key.find(".");
         size_t rf = key.rfind(".");
         if( ndot == 1 && ff != string::npos && rf != string::npos && ff == rf ){    
	     ss[key] = (it->second).as<string>();		 
         } else if ( ndot == 0 && ff == string::npos && rf == string::npos ){
             //  must avoid the vector keys : hmm a more generic way of doing this ?		 
    	     if( key != "key" && key != "val" ){          
	         ss[key] = (it->second).as<string>();		 
	     } 	 
         } 
      } 
}

void kv_dump( string msg, ssmap& ss )
{
   cout << msg << endl; 	
   ssmap::const_iterator it ;
   for( it = ss.begin() ; it != ss.end() ; ++it ) cout << "    " << it->first << " : " << it->second << endl ;   
}

void cfg_dump( sssmap& m )
{
   sssmap::const_iterator is;
   for( is = m.begin() ; is != m.end() ; ++is ){
	 string k(is->first) ;
         ssmap sm(is->second) ;	 
	 kv_dump( k , sm );
   }
}

void vm_dump( po::variables_map& vm )
{
   cout << "##vmap keys" << endl; 
   po::variables_map::const_iterator it;
   for( it = vm.begin() ; it != vm.end() ; ++it ) cout << it->first  << endl ;   
}


