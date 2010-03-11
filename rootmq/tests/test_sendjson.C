void test_sendjson()
{
   // rootmq-;rootmq-root tests/test_root2cjson.C

   gSystem->Load("librootmq");
   MQ::Create();

   gSystem->Load("libAbtDataModel");
   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;

   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   gMQ->SendJSON( ri->Class(), ri );

   exit(0) ;
}

