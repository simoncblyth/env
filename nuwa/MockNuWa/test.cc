#include "SensDet.h"

int main()
{
   G4HCofThisEvent* hce = new G4HCofThisEvent();

   SensDet* sd = new SensDet("sd");
   sd->initialize();
   sd->Initialize(hce);
}



