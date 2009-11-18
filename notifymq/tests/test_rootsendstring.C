{
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s", gSystem->GetSoExt()) );
   notifymq_init();

   const char* exchange = "t.exchange" ;
   const char* routingkey = "t.key" ;

   notifymq_sendstring( exchange ,routingkey ,"hello from test_rootsendstring.C ");
   notifymq_cleanup();
}


