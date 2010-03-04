// invoke by : 
//      notifymq-
//      notifymq-cd
//      make test_rootsendstring
//
// the kFALSE prevents starting monitor thread, just establish connection to 
// potentially remote queue and send it a message
void test_sendstring()
{	 
   gSystem->Load("libnotifymq");
   MQ::Create(kFALSE);   
   gMQ->SendString("hello from test_rootsendstring.C ");
}


