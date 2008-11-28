{ 
     // Simple geometry
      const char* esd_geom_file_name    = "http://root.cern.ch/files/alice_ESDgeometry.root";
      TFile* geom = TFile::Open(esd_geom_file_name, "CACHEREAD");
      if (!geom)
         return;
      TEveGeoShapeExtract* gse = (TEveGeoShapeExtract*) geom->Get("Gentle");
      gGeoShape = TEveGeoShape::ImportShapeExtract(gse, 0);
      geom->Close();
      delete geom;
      gEve->AddGlobalElement(gGeoShape);
}


