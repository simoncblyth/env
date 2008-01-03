// an example shows how to use TChain
// another similar method : tree->MakeSelector()
{
gROOT->Reset();

TChain t("event_tree");
t.Add("positrontable1.root");
t.Add("positrontable2.root");
t->MakeClass("myclass");
}



