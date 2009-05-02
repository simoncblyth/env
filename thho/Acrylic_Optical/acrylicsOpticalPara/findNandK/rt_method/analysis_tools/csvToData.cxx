void csvToData(void) {

    convertFormat("test","test_out");

}

void convertFormat(string inputFile, string outputFile) {


    ifstream fin;
    fin.open(inputFile.data());

    ofstream fout;
    fout.open(outputFile.data());

    string col_1, col_2;
    string comma=",";

    while(1) {
        if(!fin.good()) break;
        fin >> col_1 >> col_2;
        if(col_1 != "nm," && col_2 != "%T") {
            //col_1.replace(col_1.find(comma),comma.length(),"");
            fout << col_1 << " " << col_2 << endl;
        }
    }

    fin.close();
    fout.close();


}
