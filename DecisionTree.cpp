#include<bits/stdc++.h>
using namespace std;
const int numFeat=4;
const int maxFeatVal=5;
const double minGainVal=0.06;
const double split_ratio=0.8;
//build Node
struct Node{
    int index;
    int threshold;
    char label;
    Node* left;
    Node* right;
    Node(){
       index = -1;
       threshold = -1;
       left = NULL;
       right = NULL;
       label = '0';
    }
};
//build Data
struct Data{
  int feature[numFeat]{};
  char label='0';
      Data() {
            for(int & i : feature) {
                  i = -1;
            }
            label = '0';
      }
};

// Calculating entropy function
double getEntropy(vector<Data> data){
      //Given the data, you need to get the label, count it then return the entropy
      map<char,int> count_of;
      for (Data x:data) {
            count_of[x.label]++;
      }
      double entropy = 0;
      for (auto x_i:count_of) {
            double probability=x_i.second / double (count_of.size());
            entropy += probability * (log(probability)/log(2));
      }
      return -entropy;
}
// Information gain
double getInfoGain(vector<Data> data,vector<Data> data_left, vector<Data> data_right) {
      if (data_left.empty() || data_right.empty()) {return 0.0;}
      double ratioLeft = double(data_left.size())/double(data.size());
      double ratioRight = 1-ratioLeft;
      //cout<<"Info gain: "<<getEntropy(data)-(getEntropy(data_left)*ratioLeft+getEntropy(data_right)*ratioRight)<<endl;
      return getEntropy(data)-(getEntropy(data_left)*ratioLeft+getEntropy(data_right)*ratioRight);
}
pair<vector<Data>,vector<Data>> splitData(vector<Data> data,int index, int threshold) {
      vector<Data> data_left,data_right;
      for (Data x:data) {
            if (x.feature[index]<=threshold) {
                  data_left.push_back(x);
            }
            else data_right.push_back(x);
      }
      return make_pair(data_left,data_right);
}

char getLabel(vector<Data> &data) {
      map<char,int> count_label;
      for (Data x:data) {
            count_label[x.label]++;
      }
      int max_label = 0;
      char max_label_char = '0';
      for (auto x_i:count_label) {
            if (max_label<x_i.second) {
                  max_label_char = x_i.first;
            }
      }
      return max_label_char;
}
void buildTree(Node*& parent, vector<Data> data, int depth, int max_depth, int min_size) {
      parent = new Node();
      // Check termination conditions
      if (depth > max_depth || data.size() < min_size || getEntropy(data) == 0.0) {
            parent->label = getLabel(data);
            return;
      }

      vector<Data> data_left, data_right;
      double maxGain = -100;
      // Find best split
      for (int i = 0; i < numFeat; i++) {
            for (int j = 1; j <= maxFeatVal; j++) {
                  pair<vector<Data>, vector<Data>> split_data = splitData(data, i, j);
                  data_left = split_data.first;
                  data_right = split_data.second;
                  double gainVal = getInfoGain(data, data_left, data_right);
                  if (maxGain < gainVal) {
                        maxGain = gainVal;
                        parent->index = i;
                        parent->threshold = j;
                  }
            }
      }

      if (maxGain < minGainVal) {
            parent->label = getLabel(data);
            return;
      }

      // Split data
      pair<vector<Data>, vector<Data>> split_data = splitData(data, parent->index, parent->threshold);
      data_left = split_data.first;
      data_right = split_data.second;

      buildTree(parent->left, data_left, depth + 1, max_depth, min_size);
      buildTree(parent->right, data_right, depth + 1, max_depth, min_size);
}
char predict(Node *&parent, Data data) {
      if (parent -> left == NULL && parent -> right == NULL) {
            return parent->label;
      }
      if (data.feature[parent->index]<=parent->threshold) {
            return predict(parent->left,data);
      }
      else return predict(parent->right,data);
}
map<pair<char,char>, int> buildConfusionMatrix(vector<Data> data, Node *&root) {
      map<pair<char,char>,int> confussionMatrix;
      for (Data x:data) {
            char prediction=predict(root,x);
            if (confussionMatrix.find(make_pair(prediction,x.label)) == confussionMatrix.end()) {
                  confussionMatrix[make_pair(prediction,x.label)] = 1;
            }
            else confussionMatrix[make_pair(prediction,x.label)]++;
      }
      return confussionMatrix;

}
double CalculateF1Score(double precision, double recall) {
      if (precision + recall == 0) return 0.0;
      return 2 * (precision * recall) / (precision + recall);
}
double F1_Score(vector<Data> data, Node *&root) {
      map<pair<char,char>,int> confussionMatrix = buildConfusionMatrix(data,root);
      set<char> classes;
      for (auto x:confussionMatrix) {
            classes.insert(x.first.first);
            classes.insert(x.first.second);
      }
      map<char, double> f1Scores;
      for (auto cls:classes) {
            int TP=0, FP=0, FN=0;
            for (auto entry:confussionMatrix) {
                  char predictedClass = entry.first.first;  // Predicted class
                  char actualClass = entry.first.second;    // Actual class
                  int count = entry.second;

                  if (actualClass == cls && predictedClass == cls) {
                        TP += count;  // True positives
                  } else if (actualClass == cls && predictedClass != cls) {
                        FN += count;  // False negatives
                  } else if (actualClass != cls && predictedClass == cls) {
                        FP += count;  // False positives
                  }
            }
            double precision = (TP + FP == 0) ? 0.0 : static_cast<double>(TP) / (TP + FP);
            double recall = (TP + FN == 0) ? 0.0 : static_cast<double>(TP) / (TP + FN);
            double f1Score = CalculateF1Score(precision, recall);
            f1Scores[cls] = f1Score;
            //cout << "Class " << cls << " -> Precision: " << precision << ", Recall: " << recall << ", F1 Score: " << f1Score << endl;

      }
      double macroF1 = 0.0;
      for (const auto& entry : f1Scores) {
            macroF1 += entry.second;
      }
      macroF1 /= f1Scores.size();
      //cout << "Macro F1 Score: " << macroF1 << endl;
      return macroF1;
}
void printTree(Node *&parent, int depth) {
      if (parent -> left == NULL||depth>=100) {
            cout<<parent->label<<" "<<depth<<endl;
      }
      cout<<parent->index<<" "<<parent->threshold<<" "<<parent->label<<endl;
      printTree(parent->left,depth+1);
      printTree(parent->right,depth+1);

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
            Data d;
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
      random_shuffle(data.begin(), data.end());
}
double train_data(vector<Data> &train_data,vector<Data> &test_data, const int&maxDepth,const int&minSize) {

      Node *root = NULL;
      buildTree(root,train_data,0,maxDepth,minSize);
      return F1_Score(test_data,root);
}
tuple<double,int,int> crossValidation(vector<Data> data, const int& k) {
      shuffleData(data);
      int n = data.size();
      int foldSize=n/k;
      double best_solution=0.0;
      int best_max_depth=0,best_min_size=0;
      vector<tuple<double,int,int>> foldResult;
      vector<pair<vector<Data>,vector<Data>>> foldSet;
      for (int fold=0; fold<k; ++fold) {

            vector<Data> trainData,testData;
            int startIndex = fold * foldSize;
            int endIndex = (fold == k - 1) ? n : (fold + 1) * foldSize;
            for (int i=0; i<n;i++) {
                  if (i>=startIndex && i<endIndex) {
                        testData.push_back(data[i]);
                  }
                  else trainData.push_back(data[i]);
            }
            foldSet.push_back(make_pair(trainData,testData));
      }

      for (int  i=0;i<10;i++) {

            for (int j=0;j<10;j++) {
                  double sumKtrain=0;
                  for (auto x:foldSet) {
                        sumKtrain+=train_data(x.first,x.second,j,i);
                  }
                  if (sumKtrain>best_solution) {
                        best_solution = sumKtrain;
                        best_max_depth = j;
                        best_min_size = i;
                  }
            }

      }
      cout << "Best solution: " << best_solution << endl;
      cout << "Best max_depth: " << best_max_depth << endl;
      cout << "Best min_size: " << best_min_size << endl;
      return make_tuple(best_solution,best_max_depth,best_min_size);
}

int main(){
      vector<Data> data=read_data("train.txt");
      for (int k=7;k<=10;k++) {
            tuple<double,int,int> solution=crossValidation(data,k);
            cout<<"Best solution: "<<get<0>(solution)<<endl;
            cout<<"Best max_depth: "<<get<1>(solution)<<endl;
            cout<<"Best min_size: "<<get<2>(solution)<<endl;
      }
      return 0;
}