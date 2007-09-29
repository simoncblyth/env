{
    TFile* f = new TFile("dummy.root");
    TTree* event_tree = (TTree*)f->Get("event_tree");
    
    TString name = "fake_hit_data" ;
    TCanvas* c = new TCanvas( name , name );
    
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt");
 
     // the id specififies which FakeSD skin ...
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt:fAke.z_pmt","fAke.id==0");
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt:fAke.z_pmt","fAke.id==1");
   
    // huh everything is at id zero ???   CORRECTED NOW... CAUSED BY THE BOOLEAN BUG
    event_tree->Draw("fAke.t","fAke.id==0");
    event_tree->Draw("fAke.t","fAke.id==1");    
   
      
    //  the "weight" is an attempt at calculating the angle between the photon and the surface its crossing 
    //  ... this plot shows flip-flopping from layer to layer (with small negative contrib , presumably due to reflection back )
    event_tree->Draw("fAke.id+1:fAke.weight","eventID==1" )
    //  (Long64_t)33478

        
    // using the OpenGL viewer 
    //  set rainbow 50 level palette 
    gStyle->SetPalette(1)    
    
    // 4D view , 4th dimension is point color ... 
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt:fAke.z_pmt:fAke.id*10","eventID==1","glbox")    
        
    // zero in on the structures in the hits, using the code 
     event_tree->Draw("fAke.t:fAke.code","eventID==2&&abs(fAke.code-25000)<5000")
     event_tree->Draw("fAke.t:fAke.code","eventID==2&&abs(fAke.code-23800)<40")
        
        
     // looking at the high-timers ...  with four sensitive skins inner and outer skins on inner and outer acrylic
     //  the color corresponds to the skins    
     //
     //          blues:  0 
     //          purple: 1  
     //
     //          yellow: 2
     //             red : 3
     //   
     //    from the color patterns, clearly distinguish :
     //         * z-bouncers  
     //         * inner and outer helicals
     //   
     //   
     event_tree->Draw("fAke.code:fAke.t*100:fAke.wl*100:(fAke.id+1)*10","fAke.t>20")      
        
        
    
     // time cuts allow directionality to be seen    
      event_tree->Draw("fAke.t:fAke.z_pmt","fAke.t<10")   
    
     // time cuts are crucial to extracting info 
     // time against radius shows that ... a 14ns cut can exclude most complications 
     // of course should be direction dependeent 
       event_tree->Draw("fAke.t:TMath::Sqrt(fAke.x_pmt*fAke.x_pmt+fAke.y_pmt*fAke.y_pmt+fAke.z_pmt*fAke.z_pmt)","fAke.t<20")  
      
     // 0 and 1 almost equal ... splash back pushing up 3 cf 2    
       event_tree->Draw("fAke.id","fAke.t<14")  
     //   0 and 1 almost equal (400k) .... 2 and 3 almost equal (270k)   
       event_tree->Draw("fAke.id+1","abs(fAke.z_pmt)<1500&&fAke.t<15")
        
        
        
        
    // from legacy-blyth-1711:1719M/    
    // how are helices of hits occurring ???  TIR within the acrylic    
     event_tree->Draw("fAke.x_pmt:fAke.y_pmt:fAke.z_pmt:abs(fAke.code-23800)","eventID==2&&abs(fAke.code-23800)<40")   
    // bout equally in skins 2 and 3  
     event_tree->Draw("fAke.id","eventID==2&&abs(fAke.code-23800)<40")   
     // so occuring by internal reflection within the acrylic volume 
     // constant pz +-420 ???
     event_tree->Draw("fAke.pz:fAke.t","eventID==2&&abs(fAke.code-23800)<40")
     // clear bounce    
     event_tree->Draw("fAke.z_pmt:fAke.t","eventID==2&&abs(fAke.code-23800)<40")
        
    
    event_tree->Draw("fAke.wl:fAke.t");        // structure of 17 peaks in 370ns  ... offset of about 12 ns , spacing about 21 ns
    event_tree->Draw("int(fAke.t - 12)%21");   //   c  299.792458 mm/ns
                                               
    event_tree->Draw("fAke.wl:fAke.t","abs(fAke.z_pmt)<2000")
    event_tree->Draw("fAke.wl:fAke.t","abs(fAke.z_pmt)>2000")   // bulk of time structure coming from photons going out the top and bottom ... reflector effect ?   
    
    
    event_tree->Draw("fAke.y_pmt:fAke.x_pmt","abs(fAke.z_pmt) < 2005 " )   // circle
    event_tree->Draw("fAke.y_pmt:fAke.x_pmt","abs(fAke.z_pmt) < 2006 " )   // disc ... demo the height in z
     
         
     // peaks at 2000 and 2001 mm ... approx 10x more at 2001 ??? thho little peak effect ?      
     event_tree->Draw("TMath::Sqrt(fAke.y_pmt*fAke.y_pmt+fAke.x_pmt*fAke.x_pmt)","abs(fAke.z_pmt) < 2005 " ) 
           
    
    // 1000000 eV    
    event_tree->Draw("TMath::Sqrt(fAke.px*fAke.px+fAke.py*fAke.py+fAke.pz*fAke.pz)" ) 
    event_tree->Draw("TMath::Sqrt(fAke.x_pmt*fAke.x_pmt+fAke.y_pmt*fAke.y_pmt+fAke.z_pmt*fAke.z_pmt)" )     
        
    // position of fake hit defines a point vector from origin... find the costheta between the momentum direction and the point vector
    // to see how much outgoing cf ingoing 
    // ... at little over 50% in peak at 1 , the remainder mostly flat from -1 up to 1 with blip at 0     
    event_tree->Draw("(fAke.px*fAke.x_pmt+fAke.py*fAke.y_pmt+fAke.pz*fAke.z_pmt) / (1000000* TMath::Sqrt(fAke.x_pmt*fAke.x_pmt+fAke.y_pmt*fAke.y_pmt+fAke.z_pmt*fAke.z_pmt))" )    
    
    
    event_tree->Draw("GetNofFakeHits(-1)"); // fake hits in all FakeSD
    event_tree->Draw("GetNofFakeHits(0)");  // fake hits in FakeSD 0
    
    event_tree->Draw("fAke.code" ) ;       // zero based code, of hit numbers within an event in all FakeSD 
    event_tree->Draw("fAke.code:fAke.t");
    
    
    
    
    
    
    
    // number of fake hits in all FakeSD ...
    event_tree->Draw("GetNofFakeHits(-1)"); 
    
    // the multiple reflections, cause 
    //
    // if the fAke.code == fAke.id then its a hit from an outgoing track ... with no reflection issues
    // in which case the difference between the entries ... of 0 or 1 gives transmission and propagation between em 
    //  approx 8 percent loss
    //
    event_tree->Draw("fAke.code","fAke.code==fAke.id"); 
    
    
    event_tree->Draw("fAke.code", "fAke.code==0") ; // subtubs...   1955
    event_tree->Draw("fAke.code", "fAke.code==1")  ; // subtubs ... 1799
    
    
    event_tree->Scan("ranSeed", "GetNofFakeHits(-1)==0");
        
    /*
        ************************
        *    Row   *   ranSeed *
        ************************
        *        0 *         0 *
        *        3 *         3 *
        *        7 *         7 *
        *       14 *        14 *
        *       17 *        17 *
        *       20 *        20 *
    */
    
    
    // need the totals ... can the event_tree->Draw plot the results of a GetFakeSDHits( isd ) ????
    
    
    //event_tree->Draw("fAke.wl:fAke.t");

     // want to know the 


}