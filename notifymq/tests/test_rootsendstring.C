{
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s", gSystem->GetSoExt()) );
   notifymq_init();
   notifymq_sendstring("feed","importer","hello from test_rootsendstring.C ");
   notifymq_cleanup();
}


