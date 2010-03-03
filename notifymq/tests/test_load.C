void test_load()
{
   cout << gSystem->GetDynamicPath() << endl ;
   //gDebug = 5 ; // verbose listing of every lib and its dependencies being loaded
   //gDebug = 1 ;
   gSystem->Load("libnotifymq");
}
