{
   gSystem->Load("libnotifymq");
   MQ::Create();
   gMQ->SendString("hello from test_rootsendstring.C ");
}


