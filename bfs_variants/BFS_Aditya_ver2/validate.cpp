#include<iostream>
#include<fstream>


using namespace std;

int main (int argc, char *argv[] ) {
    //open output.txt and output2.txt and check the arrays are smae or not
    //twofiles should be taken as command line arguments
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <file1> <file2>" << endl;
        return 1;
    }
    ifstream file1(argv[1]);
    ifstream file2(argv[2]);
    if (!file1.is_open() || !file2.is_open()) {
        cout << "Error opening files" << endl;
        return 1;
    }
    int a, b;
    bool same = true;
    while (file1 >> a && file2 >> b) {
        if (a != b) {
            cout<<"at"<<endl;
            same = false;
            break;
        }
    }
    if(file1 >> a) {
        cout<<"first"<<endl;
        same = false;
    }
    else if(file2 >> b) {
        cout<<"second"<<endl;
        same = false;
    }
    if (same) {
        cout << "The files are the same" << endl;
    } else {
        cout << "The files are different" << endl;
    }
    file1.close();
    file2.close();
    return 0;
}