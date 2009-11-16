{
   gSystem->Load("$ENV_HOME/priv/lib/libprivate.so");
   private_init();
   cout << private_lookup("AMQP_SERVER") << endl ;
   private_cleanup();
}
