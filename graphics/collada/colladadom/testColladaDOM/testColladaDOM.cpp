
#include <dae.h>
#include <dom/domCOLLADA.h>

int main() {
    DAE dae;
    daeElement* root = dae.add("valid.dae");
    daeElement* asset = root->add("asset");
    daeElement* contributor = asset->add("contributor");
    daeElement* created = asset->add("created");
    daeElement* modified = asset->add("modified");
    const char* date = "2008-04-08T13:07:52-08:00";
    created->setCharData(date);
    modified->setCharData(date);
    dae.writeAll();
    return 0;
}
