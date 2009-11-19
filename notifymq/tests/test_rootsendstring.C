{
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s", gSystem->GetSoExt()) );
   MQ::Create();
   gMQ->SendString("hello from test_rootsendstring.C ");
}


