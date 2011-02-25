/*
   gdb $(which root)     
   > set args -l ~/e/AbtViz/checks/evemanager.C
   > r 
*/
{
   gSystem->Load("libEve");
   TEveManager::Create();
}
