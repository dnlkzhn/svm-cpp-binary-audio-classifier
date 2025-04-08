// Kniznice, potrebne pre implementaciu SVM
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

class SVM {
    private:
        vector<double> w;               // Vahovy vektor modelu
        double b;                       // Bias modelu  
        double lr;                      // Rychlost ucenia, definuje ako silne sa menia vahy pocas ucenia  
        double lambda_param;            // Parameter regularizacie, definuje penalizaciu vah
        int n_iters;                    // Pocet epoch ucenia
        vector<string> feature_names;   // Zoznam nazvov vlastnosti  
    
    public:
        // Konstruktor triedy, inicializuje hodnoty hyperparametrov
        SVM(double learning_rate, double lambda_param, int n_iters) {
            this->lr = learning_rate;
            this->lambda_param = lambda_param;
            this->n_iters = n_iters;
            this->b = 0;
        }
    
        // Trenovacia funkcia algoritmu SVM
        // Vstup: 
        //   - X: Matica trenovacich vzoriek
        //   - y: Vektor obsahujuci spravne triedne oznacenia
        // Vystup: 
        //   - Funkcia aktualizuje vahovy vektor `w` a bias `b` tak, aby minimalizovali chybu modelu.

        void fit(vector<vector<double>> &X, vector<int> &y) {
            int n_samples = X.size();       // Pocet trenovacich prikladov
            int n_features = X[0].size();   // Pocet vlastnosti na jeden priklad
            w.assign(n_features, 0.0);      // Inicializacia vahoveho vektora na nulove hodnoty
    
            // Hlavny cyklus iteracii ucenia
            for (int iter = 0; iter < n_iters; iter++) {
                // Prechadzanie cez vsetky vzorky v datasete
                for (int i = 0; i < n_samples; i++) {
                    double z = y[i] * (dot_product(X[i], w) + b); // Vypocet rozhodovacej hranice
                    // Spravna klasifikacia - aplikujeme iba regularizaciu
                    if (z >= 1) {
                        for (int j = 0; j < n_features; j++)
                            w[j] -= lr * (2 * lambda_param * w[j]);  // Regularizacia vah (L2 penalizacia)
                    } 
                    // Nespravna klasifikacia - aplikujeme regularizaciu a gradient descent
                    else {
                        for (int j = 0; j < n_features; j++)
                            w[j] -= lr * (2 * lambda_param * w[j] - y[i] * X[i][j]);  // Gradient descent + regularizacia
                        b -= lr * y[i] * 0.1;  // Aktualizacia biasu
                    }
                }
            }
        }
    
        // Funkcia na predikciu tried na zaklade vstupnych vlastnosti
        // Vstup:
        //   - X: Matica trestovacich vzoriek
        // Vystup:
        //   - Vektor predikovanych tried

        vector<int> predict(vector<vector<double>> &X) {
            vector<int> predictions;
            int n_samples = X.size(); // Pocet testovacich prikladov
    
            for (int i = 0; i < n_samples; i++) {
                double approx = dot_product(X[i], w) + b; // // Vypocet rozhodovacej hranice
                predictions.push_back(approx >= 0 ? 1 : -1); // Predikcia triedy na zaklade rozhodovacej hranice
            }
            return predictions;
        }
    
        // Funkcia na ulozenie parametrov modelu do suboru vo formate .txt
        // Vstup:
        //   - filename: Nazov suboru, kam sa model ulozi.
        //   - feature_names: Zoznam nazvov vlastnosti modelu.

        void save_model(const string &filename, const vector<string>& feature_names) {
            ofstream file(filename);
            int n_features = feature_names.size();
            
            if (!file.is_open()) {
                cerr << "Error: Could not open file to save model!" << endl;
                return;
            }
    
            for (int i = 0; i < n_features; i++) {
                file << feature_names[i] << " ";
            }
            file << endl;
    
            for (int i = 0; i < w.size(); i++) {
                file << w[i] << " ";
            }
            file << endl;
    
            file << b << endl;
            file.close();
            
            cout << "Model saved to " << filename << endl;
        }
    
        // Funkcia na nastavenie nazvov vlastnosti modelu
        void set_feature_names(const vector<string>& names) {
            feature_names = names;
        }
    
    private:
        // Vypocet skalarneho sucinu dvoch vektorov
        double dot_product(const vector<double> &a, const vector<double> &b) {
            double result = 0.0;
            int n_features = a.size();
            for (int i = 0; i < n_features; i++)
                result += a[i] * b[i];
            return result;
        }
    };

// Nacitanie dat z CSV suboru s nazvami vlastnosti
// Vstup:
//   - filename: Nazov CSV suboru, ktory obsahuje dataset
//   - X: Matica vlastnosti (vektor vektorov), kde kazdy riadok reprezentuje jednu vzorku
//   - y: Triedne oznacenia vzoriek (vektor)
//   - feature_names: Vektor retazcov obsahujuci nazvy vlastnosti
// Vystup:
//   - Funkcia nacita data zo suboru a naplni X, y a feature_names

void load_csv(const string &filename, vector<vector<double>> &X, vector<int> &y, vector<string> &feature_names) {
    ifstream file(filename);

    // Overenie, ci sa subor podarilo otvorit
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    bool first_line = true;  // Premenna na kontrolu hlavicky suboru

    // Citanie suboru riadok po riadku
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> features;  // Vektor na ulozenie vlastnosti vzorky
        string value;
        int label;

        // Spracovanie hlavicky CSV suboru
        if (first_line) {
            while (getline(ss, value, ',')) {
                feature_names.push_back(value);  // Ulozenie nazvov vlastnosti
            }
            feature_names.pop_back();  // Odstranenie posledneho prvku (label)
            first_line = false;  // Nastavime flag, aby sme hlavicku spracovali iba raz
            continue;
        }

        // Spracovanie datovych riadkov
        while (getline(ss, value, ',')) {
            if (ss.peek() == EOF) {
                label = stoi(value);  // Posledny stlpec obsahuje triedne oznacenie
            } else {
                features.push_back(stod(value));  // Konverzia hodnoty na double a ulozenie do vektora vlastnosti
            }
        }

        // Pridanie spracovanej vzorky do datasetu
        X.push_back(features);

        // Ulozenie triedneho oznacenia (0 -> -1, inak 1)
        y.push_back(label == 0 ? -1 : 1);
    }

    // Zatvorenie suboru po nacitani dat
    file.close();
}


// Skalovanie vlastnosti pomocou Min-Max
// Vstup:
//   - X: Matica obsahujuca vlastnosti vzoriek (kazdy vnoreny vektor predstavuje jednu vzorku)
// Vystup:
//   - Funkcia modifikuje X tak, aby boli vsetky hodnoty v rozsahu [0,1]

void scale_features(vector<vector<double>> &X) {
    int n_samples = X.size();  // Pocet vzoriek
    int n_features = X[0].size();  // Pocet vlastnosti

    vector<double> min_vals(n_features, numeric_limits<double>::max());
    vector<double> max_vals(n_features, numeric_limits<double>::lowest());

    // Hladanie minimalnych a maximalnych hodnot pre kazdu vlastnost
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            min_vals[j] = min(min_vals[j], X[i][j]);
            max_vals[j] = max(max_vals[j], X[i][j]);
        }
    }

    // Normalizacia vzoriek do rozsahu [0,1]
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            if (max_vals[j] != min_vals[j]) {
                X[i][j] = (X[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]);
            } else {
                X[i][j] = 0.0;  // Ak je max == min, nastavime hodnotu na 0
            }
        }
    }
}

// Rozdelenie datasetu na trenovaciu a testovaciu mnozinu
// Vstup:
//   - X: Matica vlastnosti vsetkych vzoriek
//   - y: Triedne oznacenia vzoriek
//   - split_ratio: Pomerny podiel dat pre trenovaciu mnozinu (napr. 0.8)
// Vystup:
//   - X_train, y_train: Trenovacia mnozina
//   - X_test, y_test: Testovacia mnozina

void split_data(const vector<vector<double>> &X, const vector<int> &y,
    vector<vector<double>> &X_train, vector<int> &y_train,
    vector<vector<double>> &X_test, vector<int> &y_test,
    double split_ratio = 0.2) 
    {

    vector<vector<double>> X_hiphop, X_classical;
    vector<int> y_hiphop, y_classical;

    // Rozdelime vzorky podla triedy
    for (int i = 0; i < X.size(); i++) {
        if (y[i] == 1) {  
            X_hiphop.push_back(X[i]);
            y_hiphop.push_back(y[i]);
        } 
        else {          
            X_classical.push_back(X[i]);
            y_classical.push_back(y[i]);
        }
    }

    // Vypocat velkosti testovacich mnozin
    int n_hiphop_test = static_cast<int>(X_hiphop.size() * split_ratio / 2); 
    int n_classical_test = static_cast<int>(X_classical.size() * split_ratio / 2); 

    // Zapis testovacich vzoriek
    for (int i = 0; i < n_hiphop_test; i++) {
        X_test.push_back(X_hiphop[i]);
        y_test.push_back(y_hiphop[i]);
    }
    for (int i = 0; i < n_classical_test; i++) {
        X_test.push_back(X_classical[i]);
        y_test.push_back(y_classical[i]);
    }

    for (int i = n_hiphop_test; i < X_hiphop.size(); i++) {
        X_train.push_back(X_hiphop[i]);
        y_train.push_back(y_hiphop[i]);
    }
    for (int i = n_classical_test; i < X_classical.size(); i++) {
        X_train.push_back(X_classical[i]);
        y_train.push_back(y_classical[i]);
    }
}


// Hodnotenie modelu pomocou roznych metrik
// Vstup:
//   - y_true: Skutocne triedne oznacenia
//   - y_pred: Predikovane triedne oznacenia modelom
// Vystup:
//   - Vypis do konzoly: Accuracy, Precision, Recall, F1 Score

void evaluate_model(const vector<int> &y_true, const vector<int> &y_pred) {
    int n_samples = y_true.size();
    int TP = 0, FP = 0, TN = 0, FN = 0;

    // Vypocet metrik na zaklade porovnania skutocnych a predikovanych hodnot
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) TP++;
        if (y_true[i] == -1 && y_pred[i] == -1) TN++;
        if (y_true[i] == -1 && y_pred[i] == 1) FP++;
        if (y_true[i] == 1 && y_pred[i] == -1) FN++;
    }

    // Vypocet metrik
    double accuracy = (double)(TP + TN) / (TP + TN + FP + FN);
    double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0;
    double recall = (TP + FN > 0) ? (double)TP / (TP + FN) : 0;
    double f1_score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

    // Vypis vysledkov
    cout << "Accuracy: " << accuracy * 100.0 << "\n";
    cout << "Precision: " << precision << "\n";
    cout << "Recall: " << recall << "\n";
    cout << "F1 Score: " << f1_score << "\n";
}


int main() {
    vector<vector<double>> X, X_train, X_test;
    vector<int> y, y_train, y_test;
    vector<string> feature_names;

    load_csv("", X, y, feature_names);
    scale_features(X);
    split_data(X, y, X_train, y_train, X_test, y_test, 0.8);
    
    SVM svm(0.01, 0.001, 50);
    svm.set_feature_names(feature_names);
    svm.fit(X_train, y_train);
    svm.save_model("", feature_names);
    
    vector<int> predictions = svm.predict(X_test);
    evaluate_model(y_test, predictions);
    
    cout << "Training completed!" << endl;
    return 0;
}
