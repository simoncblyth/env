{
   gSystem->Load("$ENV_HOME/priv/lib/libprivate.so");
   private_init();
   cout << "private_lookup(\"AMQP_SERVER\") [" << private_lookup("AMQP_SERVER") << "]" << endl ;
   cout << "private_lookup(\"AMQP_NONEXISTING\") [" << private_lookup("AMQP_NONEXISTING") << "]" << endl ;
   cout << "private_lookup_default [" << private_lookup_default("AMQP_NONEXISTING","dummy_default") << "]" << endl ;
   private_cleanup();
}
