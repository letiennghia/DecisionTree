#include<bits/stdc++.h>

#include <random>
using namespace std;
constexpr int numFeat=4;
constexpr int maxFeatVal=5;
double minGainVal=0.02;
//build Node
struct Node{
    int index;
    double threshold;
    char label;
    Node* left;
    Node* right;
    Node(){
       index = -1;
       threshold = -1;
       left = nullptr;
       right = nullptr;
       label = '0';
    }
};
//build Data
struct Data{
      char label;
      int feature[numFeat];
};

// Calculating entropy function
double getEntropy(const vector<Data>& data){
      //Given the data, you need to get the label, count it then return the entropy
      map<char,int> count_of;
      for (auto [label, feature]:data) {
            count_of[label]++;
      }
      double entropy = 0;
      for (auto [fst, snd]:count_of) {
            const double probability=static_cast<double>(snd) / data.size();
            entropy -= probability * (log(probability)/log(2));
      }
      return entropy;
}
// Information gain
double getInfoGain(const vector<Data>& data,const vector<Data>& data_left, const vector<Data>& data_right) {
      if (data_left.empty() || data_right.empty()) {return 0.0;}
      const double ratioLeft = static_cast<double>(data_left.size())/data.size();
      const double ratioRight = 1-ratioLeft;

      return getEntropy(data)-(getEntropy(data_left)*ratioLeft+getEntropy(data_right)*ratioRight);
}
// Split information
pair<vector<Data>,vector<Data>> splitData(const vector<Data>& data, const int index, const double threshold) {
      vector<Data> data_left,data_right;
      for (Data x:data) {
            if (x.feature[index]<=threshold) {
                  data_left.push_back(x);
            }
            else data_right.push_back(x);
      }
      return make_pair(data_left,data_right);
}
// Get the most number of label
char getLabel(vector<Data> &data) {
      map<char,int> count_label;
      for (Data x:data) {
            count_label[x.label]++;
      }
      int max_label = 0;
      char max_label_char = '0';
      for (auto x_i:count_label) {
            if (max_label<x_i.second) {
                  max_label = x_i.second;
                  max_label_char = x_i.first;
            }
      }
      return max_label_char;
}
void buildTree(Node*& parent, vector<Data> data, const int depth, const int max_depth, const int min_size) {
      parent = new Node();
      // Check termination conditions
      if (depth >= max_depth || data.size() <= min_size || getEntropy(data) == 0.0) {
            parent->label = getLabel(data);
            parent->left = nullptr;
            parent->right = nullptr;
            return;
      }

      vector<Data> data_left, data_right;
      double maxGain = -10;
      // Find best split
      for (int i = 0; i < numFeat; i++) {
            vector<int> feature;
            for (double j=0;j<maxFeatVal;j+=0.5){
                  auto [fst, snd] = splitData(data, i, j);
                  data_left.clear();
                  data_right.clear();
                  data_left = fst;
                  data_right = snd;
                  if (data_left.empty() || data_right.empty()) {continue;}
                  if (double gainVal = getInfoGain(data, data_left, data_right); maxGain < gainVal) {
                        maxGain = gainVal;
                        parent->index = i;
                        parent->threshold = j;
                  }
            }
            }
      if (getInfoGain(data,data_left,data_right)<minGainVal) {
            parent->label = getLabel(data);
            parent->left = nullptr;
            parent->right = nullptr;
            return;
      }

      // Split data
      auto [fst, snd] = splitData(data, parent->index, parent->threshold);
      data_left = fst;
      data_right = snd;
      //Build children
      buildTree(parent->left, data_left, depth + 1, max_depth, min_size);
      buildTree(parent->right, data_right, depth + 1, max_depth, min_size);
      if(parent -> left -> label != '0' && parent -> left -> label == parent -> right -> label){
            parent->label = parent->left->label;
            parent->left = nullptr;
            parent->right = nullptr;
      }
}
//prediction
char predict(Node *&parent, Data data) {
      if (parent -> left == nullptr) {
            return parent->label;
      }
      if (data.feature[parent->index]<=parent->threshold) {
            return predict(parent->left,data);
      }
      else return predict(parent->right,data);
}
//CalcualteF1Score
double CalculateF1Score(double precision, double recall) {
      if (precision + recall == 0) return 0.0;
      return 2 * (precision * recall) / (precision + recall);
}

double F1_Score(const vector<Data>& data, Node *&root) {
    // Step 1: Identify all unique classes
    set<char> classes;
    for (const auto& d : data) {
        classes.insert(d.label); // Add actual labels
    }

    // Variables to accumulate TP, FP, FN for micro F1
    int totalTP = 0, totalFP = 0, totalFN = 0;

    // Step 2: Calculate F1 score for each class
    map<char, double> f1Scores;
    for (auto cls : classes) {
        int TP = 0, FP = 0, FN = 0;

        // Step 3: Loop over the dataset and calculate TP, FP, FN
        for (const auto& d : data) {
            char predictedLabel = predict(root, d);  // Predict the label using the tree

            if (d.label == cls && predictedLabel == cls) {
                TP++;  // True Positive
            } else if (d.label != cls && predictedLabel == cls) {
                FP++;  // False Positive
            } else if (d.label == cls && predictedLabel != cls) {
                FN++;  // False Negative
            }
        }

        // Step 4: Calculate precision and recall for this class
        double precision = (TP + FP == 0) ? 0.0 : static_cast<double>(TP) / (TP + FP);
        double recall = (TP + FN == 0) ? 0.0 : static_cast<double>(TP) / (TP + FN);

        // Calculate F1 score for this class
        double f1Score = CalculateF1Score(precision, recall);
        f1Scores[cls] = f1Score;

        // Add to micro F1 totals
        totalTP += TP;
        totalFP += FP;
        totalFN += FN;

        // Optionally, print the precision, recall, and F1 score for this class
        cout << "Class " << cls << " -> Precision: " << precision << ", Recall: " << recall << ", F1 Score: " << f1Score << endl;
    }

    // Step 5: Calculate macro F1 score (average of F1 scores of all classes)
    double macroF1 = 0.0;
    for (const auto& [cls, f1Score] : f1Scores) {
        macroF1 += f1Score;
    }
    macroF1 /= f1Scores.size();  // Average F1 score

    // Step 6: Calculate micro F1 score
    double microPrecision = (totalTP + totalFP == 0) ? 0.0 : static_cast<double>(totalTP) / (totalTP + totalFP);
    double microRecall = (totalTP + totalFN == 0) ? 0.0 : static_cast<double>(totalTP) / (totalTP + totalFN);
    double microF1 = CalculateF1Score(microPrecision, microRecall);

    // Output F1 scores
    cout << "Macro F1 Score: " << macroF1 << endl;
    cout << "Micro F1 Score: " << microF1 << endl;

    return macroF1;  // Returning macro F1 score
}
vector<Data> read_data(const string &filename) {
      vector<Data> data;
      ifstream file(filename);
      if (!file.is_open()) {
            cout<<"File Not Found"<<endl;
            return data;
      }
      string line;
      while (getline(file,line)) {

            vector<int> features;
            Data d{};
            for (char c:line) {

                  if (c==',') continue;
                  if (c>='A' && c<='Z') {
                        //cout<<c<<" ";
                        d.label = c;
                  }
                  else if (c>='0' && c<='9') {
                        //cout<<c<<" ";
                        features.push_back(c-'0');
                  }

            }
            for (int i=0;i<features.size();i++) {
                  d.feature[i] = features[i];
            }
            data.push_back(d);
            //cout<<d.label<<" "<<d.feature[0]<<" "<<d.feature[1]<<" "<<d.feature[2]<<" "<<d.feature[3]<<endl;
      }
      return data;
}
void shuffleData(vector<Data> &data) {
      shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));
}
double train_data(Node *&root,const vector<Data> &train_data, const vector<Data> &test_data, const int&maxDepth,const int&minSize) {


      buildTree(root,train_data,0,maxDepth,minSize);
      //printTree(root,0);
      return F1_Score(test_data,root);
}
tuple<double, int, int,vector<Data>,vector<Data>> crossValidation(vector<Data> data) {
      shuffleData(data);  // Shuffle the data before splitting
      Node *root = nullptr;
      const int n = data.size();
      double best_solution = 0.0;
      int best_max_depth = 1, best_min_size = 1;
      double bestF1Score = 0.0;
      const size_t splitIndex=n*0.8;
      const vector<Data> trainData(data.begin(), data.begin()+splitIndex);
      const vector<Data> testData(data.begin()+splitIndex,data.end());
      for (int i=1;i<=20;i++) {
            for (int j=1;j<=20;j++)
                  if (bestF1Score<train_data(root,trainData,testData,i,j)) {
                        bestF1Score = train_data(root,trainData,testData,i,j);
                        best_max_depth = i;
                        best_min_size = j;
                  }

      }
      cout << "Best F1: " << bestF1Score << endl;
      cout << "Best max_depth: " << best_max_depth << endl;
      cout << "Best min_size: " << best_min_size << endl;
      return make_tuple(bestF1Score, best_max_depth, best_min_size,trainData,testData);
} //Dataset - Classifier Info


int main(){
      vector<Data> data=read_data("Data/");
      vector<Data> test_data=read_data("Data/");
      Node *root=nullptr;
      double maxF1Score=0.0, max_depth=0, min_size=0;
      vector<Data> goodTrainData, goodTestData;
      for (int i=0;i<10;i++) {
            if (tuple<double,int,int,vector<Data>,vector<Data>> result=crossValidation(data); maxF1Score<get<0>(result)) {
                  maxF1Score = get<0>(result);
                  max_depth = get<1>(result);
                  min_size = get<2>(result);
                  goodTrainData = get<3>(result);
                  goodTestData=get<4>(result);
            }
      }
      buildTree(root,goodTrainData,0,max_depth,min_size);// After this we have one Classifier

      int cnt=1;
      cout<<"Max F1: " << maxF1Score << endl;
      freopen("Predict_labels.csv","w",stdout);
      cout<<"ID,Label"<<endl;
      for (Data t:test_data) {
            cout<<cnt<<","<<predict(root,t)<<endl;
            cnt++;
      }



      return 0;


}