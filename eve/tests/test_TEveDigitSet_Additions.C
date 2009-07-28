
{

   gROOT->ProcessLine(".L TEveDigitSet_Additions.cxx+");

   TRandom r(0);
   TEveQuadSet* q = new TEveQuadSet("RectangleXY");
   q->Print();
   q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);

   Int_t num = 10 ;
   for (Int_t i=0; i<num; ++i)
   {
      q->AddQuad(r.Uniform(-10, 9), r.Uniform(-10, 9), 0, r.Uniform(0.2, 1), r.Uniform(0.2, 1));
      q->QuadValue(100+i);
      q->QuadId(new TNamed(Form("QuadIdx %d", i), "TNamed assigned to a quad as an indentifier."));
   }
   q->RefitPlex();

   cout << "TEveDigitSet_GetValue ... " ;
   for (Int_t i=0; i<num; ++i) cout << TEveDigitSet_GetValue( q , i ) << " " ;
   cout << endl ;

   cout << "TEveDigitSet_SetValue ... " ;
   for (Int_t i=0; i<num; ++i) TEveDigitSet_SetValue( q , i , i + 200 )  ;
   for (Int_t i=0; i<num; ++i) cout << TEveDigitSet_GetValue( q , i ) << " " ;
   cout << endl ;

   cout << "TEveDigitSet_PrintValue ... " << endl ;
   for (Int_t i=0; i<num; ++i) TEveDigitSet_PrintValue( q , i )  ;


}


