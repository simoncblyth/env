
#include "root2cjson.h"

#include <iostream>   
using namespace std;   

#include "TObject.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TMethodCall.h"
#include "TList.h"

cJSON* root2cjson( TClass* kls,  TObject* obj )
{
   TIter next(kls->GetListOfAllPublicDataMembers());
   cJSON* o = cJSON_CreateObject(); 
   char* ss[512]  ;  
   Long_t ll ;
   Double_t dd ; 
   const char* params = NULL ;

   TDataMember* dm = NULL ;
   while((dm = (TDataMember*)next())){
       TString name = dm->GetName() ;
       TString type = dm->GetTrueTypeName();
       Int_t dim = dm->GetArrayDim();
       Int_t max = dm->GetMaxIndex(0); 
       TString smry = Form("name %s type %s dim %d max %d" , name.Data(), type.Data(), dim , max ); 

       if(name.BeginsWith("f")){
           //cout << smry << endl ;
           name =  name(1,name.Length()-1);  // trim the "f"
           TMethodCall* getter = dm->GetterMethod(kls);

           if( type == "char" && dim == 1  && max > 0 && max < 512  ){  // character array 
                getter->Execute( obj , "" , ss ); 
                cJSON_AddItemToObject(o, name.Data(), cJSON_CreateString(*ss) );

           } else if ( type == "unsigned int" && dim == 0 && max == -1  ){  // single unsigned int
                getter->Execute( obj , params , ll ); 
                cJSON_AddItemToObject(o, name.Data(), cJSON_CreateNumber(ll) );

           } else if ( type == "float"        && dim == 0 && max == -1  ){  // single float
                getter->Execute( obj , params , dd ); 
                cJSON_AddItemToObject(o, name.Data(), cJSON_CreateNumber(dd) );

           } else {
                cJSON_AddItemToObject(o, name.Data(), cJSON_CreateString(Form("SKIPPED %s", smry.Data()) ));
           }


       }    // fMembers 
   }        // over members

   //cout << cJSON_Print(o) << endl ;
   return o ;
}


