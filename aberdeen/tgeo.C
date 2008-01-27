
{
    TGeoManager* tgm = TGeoManager::Import("Aberdeen_World.root");  


    TGeoVolume* tv = gGeoManager->GetTopVolume();
    TGeoNode* tn = gGeoManager->GetTopNode();
    
    tv->Draw("ogle");

    // see $ROOTSYS/etc/system.rootrc    for 3D viewer control 
    //
    //  Issues:
    //     * using OpenGL viewer causes Bus error on Darwin

}




