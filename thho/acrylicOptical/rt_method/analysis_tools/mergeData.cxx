void mergeData(void) {

    FILE* pipe = gSystem->OpenPipe("ls *.csv" , "r" );
    TString finname;
    TString foutname;
    while(finname.Gets(pipe)) {
        foutname = "out.dat";
        cout << finname << " " << foutname << endl;
        convertFormat(finname,foutname);
    }

    gSystem->Exit( gSystem->ClosePipe( pipe ));



}

void convertFormat(TString inputFile, TString outputFile) {


    ifstream fin;
    fin.open(inputFile.Data());

    ofstream fout;
    fout.open(outputFile.Data(), ios_base::app);

    TString col_1, col_2;


    while(1) {
        fin >> col_1 >> col_2;
        if(!fin.good()) break;
        if((col_1 != "nm,") && ((col_2 != "%T") || (col_2 != "%R") )) {
            Int_t colEnd = col_1.Length();
            colEnd -= 1;
            col_1.Remove(colEnd);
            fout << col_1 << " " << col_2 << endl;
        }
    }


    fin.close();
    fout.close();


}
