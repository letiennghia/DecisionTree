#include<bits/stdc++.h>
#include <random>
using namespace std;

const int numFeat = 4;
const int maxFeatVal = 5;
const double minGainVal = 0.06;

// Build Node
struct Node {
    int index;
    int threshold;
    char label;
    Node* left;
    Node* right;

    Node() {
        index = -1;
        threshold = -1;
        left = NULL;
        right = NULL;
        label = '0';
    }
};

// Build Data
struct Data {
    char label;
    int feature[numFeat];
};

// Calculating entropy function
double getEntropy(vector<Data> data) {
    // Given the data, you need to get the label, count it then return the entropy
    map<char, int> count_of;
    for (Data x : data) {
        count_of[x.label]++;
    }
    double entropy = 0;
    for (auto x_i : count_of) {
        double probability = (double)x_i.second / data.size();
        entropy -= probability * (log(probability) / log(2));
    }
    return entropy;
}

// Information gain
double getInfoGain(vector<Data> data, vector<Data> data_left, vector<Data> data_right) {
    if (data_left.empty() || data_right.empty()) { return 0.0; }
    double ratioLeft = (double)data_left.size() / data.size();
    double ratioRight = 1 - ratioLeft;

    return getEntropy(data) - (getEntropy(data_left) * ratioLeft + getEntropy(data_right) * ratioRight);
}

pair<vector<Data>, vector<Data>> splitData(vector<Data> data, int index, int threshold) {
    vector<Data> data_left, data_right;
    for (Data x : data) {
        if (x.feature[index] <= threshold) {
            data_left.push_back(x);
        }
        else data_right.push_back(x);
    }
    return make_pair(data_left, data_right);
}

char getLabel(vector<Data>& data) {
    map<char, int> count_label;
    for (Data x : data) {
        count_label[x.label]++;
    }
    int max_label = 0;
    char max_label_char = '0';
    for (auto x_i : count_label) {
        if (max_label < x_i.second) {
            max_label = x_i.second;
            max_label_char = x_i.first;
        }
    }
    return max_label_char;
}

void buildTree(Node*& parent, vector<Data> data, int depth, int max_depth, int min_size) {
    parent = new Node();
    // Check termination conditions
    if (depth >= max_depth || data.size() <= min_size || getEntropy(data) == 0.0) {
        parent->label = getLabel(data);
        return;
    }

    vector<Data> data_left, data_right;
    double maxGain = -100;
    // Find best split
    for (int i = 0; i < numFeat ; i++) {
        for (int j = 0; j <= maxFeatVal; j++) {
            pair<vector<Data>, vector<Data>> split_data = splitData(data, i, j);
            data_left = split_data.first;
            data_right = split_data.second;
            if (data_left.empty() || data_right.empty()) { continue; }
            double gainVal = getInfoGain(data, data_left, data_right);
            if (maxGain < gainVal) {
                maxGain = gainVal;
                parent->index = i;
                parent->threshold = j;
            }
        }
    }

    // Split data
    pair<vector<Data>, vector<Data>> split_data = splitData(data, parent->index, parent->threshold);
    data_left = split_data.first;
    data_right = split_data.second;

    buildTree(parent->left, data_left, depth + 1, max_depth, min_size);
    buildTree(parent->right, data_right, depth + 1, max_depth, min_size);

    if (parent->left->label != '0' && parent->left->label == parent->right->label) {
        parent->label = parent->left->label;
        parent->left = NULL;
        parent->right = NULL;
    }
}

char predict(Node*& parent, Data data) {
    if (parent->left == NULL) {
        return parent->label;
    }
    if (data.feature[parent->index] <= parent->threshold) {
        return predict(parent->left, data);
    }
    else return predict(parent->right, data);
}

map<pair<char, char>, int> buildConfusionMatrix(vector<Data> data, Node*& root) {
    map<pair<char, char>, int> confusionMatrix;
    for (Data x : data) {
        char prediction = predict(root, x);
        if (confusionMatrix.find(make_pair(prediction, x.label)) == confusionMatrix.end()) {
            confusionMatrix[make_pair(prediction, x.label)] = 1;
        }
        else confusionMatrix[make_pair(prediction, x.label)]++;
    }
    char val[3] = { 'L', 'R', 'B' };
    for (char c1 : val) {
        for (char c2 : val) cout << confusionMatrix[make_pair(c1, c2)] << " ";
        cout << endl;
    }
    cout << endl;
    return confusionMatrix;
}

double CalculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2 * (precision * recall) / (precision + recall);
}

double F1_Score(vector<Data> data, Node*& root) {
    map<pair<char, char>, int> confusionMatrix = buildConfusionMatrix(data, root);
    set<char> classes;
    for (auto x : confusionMatrix) {
        classes.insert(x.first.first);
        classes.insert(x.first.second);
    }
    map<char, double> f1Scores;
    for (auto cls : classes) {
        int TP = 0, FP = 0, FN = 0;
        for (auto entry : confusionMatrix) {
            char predictedClass = entry.first.first;  // Predicted class
            char actualClass = entry.first.second;    // Actual class
            int count = entry.second;

            if (actualClass == cls && predictedClass == cls) {
                TP += count;  // True positives
            }
            else if (actualClass == cls && predictedClass != cls) {
                FN += count;  // False negatives
            }
            else if (actualClass != cls && predictedClass == cls) {
                FP += count;  // False positives
            }
        }
        double precision = (TP + FP == 0) ? 0.0 : static_cast<double>(TP) / (TP + FP);
        double recall = (TP + FN == 0) ? 0.0 : static_cast<double>(TP) / (TP + FN);
        double f1Score = CalculateF1Score(precision, recall);
        f1Scores[cls] = f1Score;
    }
    double macroF1 = 0.0;
    for (const auto& entry : f1Scores) {
        macroF1 += entry.second;
    }
    macroF1 /= f1Scores.size();
    return macroF1;
}

void printTree(Node*& parent, int depth) {
    if (parent == NULL) { return; }
    if (parent->left == NULL || depth >= 100) {
        cout << parent->label << " " << depth << endl;
    }
    cout << "L2" << endl;
    cout << parent->index << " " << parent->threshold << " " << parent->label << endl;
    cout << "left" << endl;
    printTree(parent->left, depth + 1);
    printTree(parent->right, depth + 1);
}

vector<Data> read_data(const string& filename) {
    vector<Data> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "File Not Found" << endl;
        return data;
    }
    string line;
    while (getline(file, line)) {
        vector<int> features;
        Data d;
        int feature_count = 0;

        // Parse each line to extract features and label
        for (char c : line) {
            if (c == ',') continue;  // Skip commas
            if (c >= 'A' && c <= 'Z') {
                d.label = c;  // Set the label (assuming a single letter label)
            }
            else if (c >= '0' && c <= '9') {
                if (feature_count < numFeat) {
                    features.push_back(c - '0');  // Convert char to int for features
                    feature_count++;
                }
            }
        }

        // Assign features to the Data struct
        for (int i = 0; i < feature_count; i++) {
            d.feature[i] = features[i];
        }

        // Push the data point to the dataset
        data.push_back(d);
    }

    return data;
}

void shuffleData(vector<Data>& data) {
    shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));
}

double train_data(vector<Data>& train_data, vector<Data>& test_data, const int& maxDepth, const int& minSize) {
    Node* root = NULL;
    buildTree(root, train_data, 0, maxDepth, minSize);
    return F1_Score(test_data, root);
}

// Main function with K-fold Cross Validation
int main() {
    vector<Data> data = read_data("train.txt");

    int foldCount = 5;
    int maxDepth = 5;
    int minSize = 10;

    vector<double> scores;
    for (int i = 0; i < foldCount; i++) {
        vector<Data> trainData, testData;
        shuffleData(data);

        // Split data into k-fold
        int foldSize = data.size() / foldCount;
        int startIdx = i * foldSize;
        int endIdx = (i + 1) * foldSize;

        testData.insert(testData.end(), data.begin() + startIdx, data.begin() + endIdx);
        trainData.insert(trainData.end(), data.begin(), data.begin() + startIdx);
        trainData.insert(trainData.end(), data.begin() + endIdx, data.end());

        // Train and get F1 score
        double score = train_data(trainData, testData, maxDepth, minSize);
        scores.push_back(score);
    }

    // Calculate average F1 score
    double totalScore = 0.0;
    for (double score : scores) {
        totalScore += score;
    }
    double averageScore = totalScore / foldCount;
    cout << "Average F1 Score: " << averageScore << endl;

    return 0;
}
