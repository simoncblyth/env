{
  // started from 
  //   http://root.cern.ch/root/html/examples/formula1.C.html
  //
  //
  // To see the graphics output of this macro, click  here. 
  //
  gROOT->Reset();
  c1 = new TCanvas("c1","Example with Formula",200,10,700,500);
  //
  // We create a formula object and compute the value
  // of this formula
  // for two different values of the x variable.
  //
  form1 = new TFormula("form1","sqrt(abs(x))"); form1->Eval(2); form1->Eval(-45);
  //
  // Create a one dimensional
  // function and draw it
  //
  fun1 = new TF1("fun1","abs(sin(x)/x)",0,10);
  c1->SetGridx();
  c1->SetGridy();
  fun1->Draw();
  c1->Update();
  //
  // Before
  // leaving
  // this
  // demo,
  // we
  // print
  // the
  // list
  // of
  // objects
  // known
  // to
  // ROOT
  //
  gObjectTable->Print();


  TFile f("formula1.root", "recreate" );
  c1->Write();
  f.Close();
}
