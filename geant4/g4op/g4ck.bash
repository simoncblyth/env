# === func-gen- : geant4/g4op/g4ck fgp geant4/g4op/g4ck.bash fgn g4ck fgh geant4/g4op src base/func.bash
g4ck-source(){   echo ${BASH_SOURCE} ; }
g4ck-edir(){ echo $(dirname $(g4ck-source)) ; }
g4ck-ecd(){  cd $(g4ck-edir); }
g4ck-dir(){  echo $LOCAL_BASE/env/geant4/g4op/g4ck ; }
g4ck-cd(){   cd $(g4ck-dir); }
g4ck-vi(){   vi $(g4ck-source) ; }
g4ck-env(){  elocal- ; }
g4ck-usage(){ cat << EOU


* https://uspas.fnal.gov/materials/10MIT/Review_of_Relativity.pdf

   


* https://arxiv.org/pdf/1206.5530.pdf

Calculation of the Cherenkov light yield from low energetic secondary particles
accompanying high-energy muons in ice and water with Geant4 simulations

* :google:`cerenkov photon yield formula`

* https://user-web.icecube.wisc.edu/~tmontaruli/801/lect10.pdf


::

    epsilon:issues blyth$ g4-cls G4Cerenkov
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02
    vi -R source/processes/electromagnetic/xrays/include/G4Cerenkov.hh source/processes/electromagnetic/xrays/src/G4Cerenkov.cc
    2 files to edit

    596 // GetAverageNumberOfPhotons
    597 // -------------------------
    598 // This routine computes the number of Cerenkov photons produced per
    599 // GEANT-unit (millimeter) in the current medium.
    600 //             ^^^^^^^^^^
    601 
    602 G4double
    603   G4Cerenkov::GetAverageNumberOfPhotons(const G4double charge,
    604                                         const G4double beta,
    605                             const G4Material* aMaterial,
    606                             G4MaterialPropertyVector* Rindex) const
    607 {
    608   const G4double Rfact = 369.81/(eV * cm);

    ///   fine_structure_const_over_hbarc 

    609 
    610   if(beta <= 0.0)return 0.0;
    611 
    612   G4double BetaInverse = 1./beta;
    613 
    614   // Vectors used in computation of Cerenkov Angle Integral:
    615   //    - Refraction Indices for the current material
    616   //    - new G4PhysicsOrderedFreeVector allocated to hold CAI's
    617 
    618   G4int materialIndex = aMaterial->GetIndex();
    619 
    620   // Retrieve the Cerenkov Angle Integrals for this material  
    621 
    622   G4PhysicsOrderedFreeVector* CerenkovAngleIntegrals =
    623              (G4PhysicsOrderedFreeVector*)((*thePhysicsTable)(materialIndex));
    624 
    625   if(!(CerenkovAngleIntegrals->IsFilledVectorExist()))return 0.0;
    626 
    627   // Min and Max photon energies 
    628   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    629   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    630 
    631   // Min and Max Refraction Indices 
    632   G4double nMin = Rindex->GetMinValue();
    633   G4double nMax = Rindex->GetMaxValue();
    634 
    635   // Max Cerenkov Angle Integral 
    636   G4double CAImax = CerenkovAngleIntegrals->GetMaxValue();
    637 
    638   G4double dp, ge;
    639 
    640   // If n(Pmax) < 1/Beta -- no photons generated 
    641 
    642   if (nMax < BetaInverse) {
    643      dp = 0.0;
    644      ge = 0.0;
    645   } 
    646 
    647   // otherwise if n(Pmin) >= 1/Beta -- photons generated  
    648 
    649   else if (nMin > BetaInverse) {
    650      dp = Pmax - Pmin;
    651      ge = CAImax;
    652   } 
    653 
    654   // If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
    655   // we need to find a P such that the value of n(P) == 1/Beta.
    656   // Interpolation is performed by the GetEnergy() and
    657   // Value() methods of the G4MaterialPropertiesTable and
    658   // the GetValue() method of G4PhysicsVector.  
    659 
    660   else {
    661      Pmin = Rindex->GetEnergy(BetaInverse);
    662      dp = Pmax - Pmin;
    663 
    664      // need boolean for current implementation of G4PhysicsVector
    665      // ==> being phased out
    666      G4bool isOutRange;
    667      G4double CAImin = CerenkovAngleIntegrals->GetValue(Pmin, isOutRange);
    668      ge = CAImax - CAImin;
    669 
    670      if (verboseLevel>0) {
    671         G4cout << "CAImin = " << CAImin << G4endl;
    672         G4cout << "ge = " << ge << G4endl;
    673      }
    674   }
    675    
    676   // Calculate number of photons 
    677   G4double NumPhotons = Rfact * charge/eplus * charge/eplus *
    678                                  (dp - ge * BetaInverse*BetaInverse);
    679 
    680   return NumPhotons;
    681 }




::

    UseGeant4::physical_constants
                                                   eV 1e-06
                                                   cm 10
                                 fine_structure_const 0.00729735
                        one_over_fine_structure_const 137.036
              fine_structure_const_over_hbarc*(eV*cm) 369.81021
                      fine_structure_const_over_hbarc 36981020.84589
                           Rfact =  369.81/(eV * cm)  36981000.00000
                                                eplus 1.00000
    epsilon:UseGeant4 blyth$ 



EOU
}
g4ck-get(){
   local dir=$(dirname $(g4ck-dir)) &&  mkdir -p $dir && cd $dir

}
