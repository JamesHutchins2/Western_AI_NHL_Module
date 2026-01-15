# NHL Draft Prediction - Machine Learning Module

A comprehensive machine learning module for predicting NHL first-round draft picks based on junior hockey statistics. This standalone module provides the core ML models and chatbot functionality used in the full-stack NHL Draft Prediction System.

## Project Overview

This module serves as the machine learning engine for NHL draft prediction, featuring:
- **Complete ML Pipeline**: Data preprocessing, feature engineering, model training, and evaluation
- **6 Production Models**: Ensemble methods and deep learning models saved for deployment
- **Intelligent Chatbot**: NLP-powered conversational interface for predictions
- **Model Comparison**: Comprehensive evaluation of multiple classification approaches

## Module Purpose

This is the **core machine learning component** extracted from the full-stack application (NHLfinalBuild). It can be used as:
- Standalone prediction module
- Training pipeline for model updates
- Research codebase for ML experimentation
- Chatbot module for integration into other applications

## Project Structure

```
Western_AI_NHL_Module/
├── NHL.ipynb                          # Complete ML pipeline (1152 lines)
├── NHL_Chat_Bot_Master/
│   ├── NHL.ipynb                      # Chatbot-focused notebook
│   ├── test_main.py                   # Core chatbot logic
│   ├── user_functions.py              # NLP preprocessing utilities
│   ├── responses.py                   # Response templates
│   ├── bag_dtree_model.joblib         # Bagged Decision Tree
│   ├── bag_lr_model.joblib            # Bagged Logistic Regression
│   ├── bag_svm_model.joblib           # Bagged SVM (best: 81.44%)
│   ├── random_forest_model.joblib     # Random Forest
│   ├── simple_MLP_model.h5            # Simple Neural Network
│   ├── complex_MLP_model.h5           # Tuned Neural Network
│   ├── scaler.joblib                  # StandardScaler for features
│   ├── player_names.csv               # Player name lookup
│   └── stats.csv                      # NHL statistics dataset
└── README.md
```

## Machine Learning Pipeline

### 1. Data Collection & Merging
**Two Primary Data Sources**:
- **NHL Draft Data** (1963-2022): Draft picks, teams, positions, overall pick numbers
- **Elite Prospects Data**: Junior hockey statistics (NCAA, OHL, WHL, QMJHL, etc.)

**Merging Strategy**:
```python
# Handle duplicate player names by temporal sorting
junior_data = junior_data.sort_values(by='start_year', ascending=False)
junior_data.drop_duplicates(subset='player', keep='first', inplace=True)

df = pd.merge(df, junior_data, on='player')
```

### 2. Data Cleaning & Preprocessing

**League Filtering**:
```python
junior_leagues = ['NCAA', 'USHL', 'WHL', 'BCHL', 'OHL', 'QMJHL', 
                  'AJHL', 'MJHL', 'SIJHL', 'SJHL', 'CCHL', 'NOJHL', 
                  'OJHL', 'MJAHL', 'QJAAAHL']
```

**Position Standardization**:
- Consolidated variations (W→RW, F→C, Centr→C)
- One-hot encoding for final model input

**Outlier Removal**:
- Z-score analysis (threshold = 2) on Games Played and Points
- Removed first-round picks with <40 points (data errors)
- Removed non-first-round picks with >100 points (data errors)
- Temporal filter: Only post-1990 data (consistent game rules)

**Final Dataset**:
- **Target Variable**: `first_round` (1 = first 32 picks, 0 = rounds 2-7)
- **Class Distribution**: 9% first-round picks, 91% other rounds
- **Features**: Goals, Assists, Games Played, Penalty Minutes, Position

### 3. Feature Engineering & Selection

**Feature Selection Methods**:

#### Principal Component Analysis (PCA)
```python
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
info_prop = eigenvalues / eigenvalues.sum()
cum_info_prop = np.cumsum(info_prop)

# Result: 6 features explain 95% of variance
```

#### Sequential Feature Selection (SFS/SBS)
```python
from mlxtend.feature_selection import SequentialFeatureSelector

# Forward selection with floating
sfs = SFS(lr, k_features=6, forward=True, floating=True, 
          scoring='accuracy', cv=0)
sfs.fit(x, y)

# Accuracy: 0.73+ with 6 features
```

#### Recursive Feature Elimination (RFE)
```python
rfe = RFE(estimator=lr, n_features_to_select=6)
rfe.fit(x, y)

selected_features = ['G', 'A', 'GP', 'PIM', 'position_D', 'position_LW']
# Accuracy: 0.7304
```

**Selected Features**:
1. **G** - Goals
2. **A** - Assists
3. **GP** - Games Played
4. **PIM** - Penalty Minutes
5. **position_D** - Defenseman indicator
6. **position_LW** - Left Wing indicator

### 4. Model Development & Evaluation

#### Traditional Machine Learning Models

**Logistic Regression**
```python
lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)
# Accuracy: 73.04%
```

**K-Nearest Neighbors**
```python
k_neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
k_neigh.fit(x_train, y_train)
# Accuracy: 89.7% (overfitted to negative class - discarded)
```

**Decision Tree**
```python
dTree = DecisionTreeClassifier(max_depth=7, class_weight='balanced')
dTree.fit(x_train, y_train)
# Accuracy: 70.60%
```

**Support Vector Machine**
```python
svc = SVC(kernel="rbf", gamma=1, C=100, class_weight='balanced')
svc.fit(x_train, y_train)
# Accuracy: 73.58%
```

#### Ensemble Methods

**Random Forest**
```python
rf = RandomForestClassifier(bootstrap=True, max_depth=7, 
                            n_estimators=10, class_weight='balanced')
rf.fit(x_train, y_train)
# Accuracy: 73.58% (improved DTree by 3%)
```

**Bagging - Decision Tree**
```python
bagging_DTree = BaggingClassifier(
    DecisionTreeClassifier(max_depth=7, class_weight='balanced'),
    n_estimators=100
)
bagging_DTree.fit(x_train, y_train)
# Accuracy: 74.80% (improved by 4.2%)
```

**Bagging - SVM** ⭐ **Best Model**
```python
bagging_SVM = BaggingClassifier(
    SVC(kernel="rbf", gamma=1, C=10, class_weight='balanced'),
    n_estimators=100
)
bagging_SVM.fit(x_train, y_train)
# Accuracy: 81.44% (improved by ~10%)
# Recall: 55% for first-round class
```

**Bagging - Logistic Regression**
```python
bagging_lr = BaggingClassifier(
    LogisticRegression(class_weight='balanced'),
    n_estimators=10
)
bagging_lr.fit(x_train, y_train)
# Accuracy: 73.44%
```

**AdaBoost** (Adaptive Boosting)
```python
ada_classifier = AdaBoostClassifier(
    base_estimator=decision_stump,
    n_estimators=5
)
ada_classifier.fit(x_train, y_train)
# Performance decreased - overfitting issues
```

**Gradient Boosting**
```python
grad_classifier = GradientBoostingClassifier(n_estimators=10)
grad_classifier.fit(x_train, y_train)
# 91.19% accuracy but extremely imbalanced (discarded)
```

#### Deep Learning Models

**Simple MLP**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])
# Accuracy: ~75%
```

**Complex MLP** (Hyperparameter Tuned)
```python
def create_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(hp.Int('dense1_units', 32, 256, 32), 
                           activation='relu', input_shape=(6,)))
    model.add(layers.Dropout(hp.Float('dropout1', 0.1, 0.5, 0.1)))
    model.add(layers.Dense(hp.Int('dense2_units', 32, 256, 32), 
                           activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout2', 0.1, 0.5, 0.1)))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

tuner = Hyperband(create_model, objective='val_accuracy', 
                  max_epochs=20, factor=3)
tuner.search(x_train, y_train, validation_data=(x_val, y_val))
# Accuracy: ~78%
```

### 5. Hyperparameter Tuning

**SVM Grid Search**
```python
parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001]}
clf = GridSearchCV(svc, parameters, refit=True, 
                   verbose=2, scoring='accuracy')
# Best: C=100, gamma=1
```

**Decision Tree Grid Search**
```python
parameters = {'criterion': ['gini', 'entropy'], 
              'max_depth': [1, 2, 3, 4, 5, 6, 7]}
clf = GridSearchCV(dTree, parameters, refit=True, verbose=2)
# Best: criterion='gini', max_depth=7
```

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Logistic Regression | 73.04% | - | - | - | Baseline |
| K-Nearest Neighbors | 89.7% | - | Low | - | ❌ Discarded (overfitted) |
| Decision Tree | 70.60% | - | - | - | Baseline |
| SVM (tuned) | 73.58% | - | - | - | Baseline |
| Random Forest | 73.58% | - | - | - | +3% vs DTree |
| **Bagged SVM** | **81.44%** | **High** | **55%** | **High** | ✅ **Best** |
| Bagged Decision Tree | 74.80% | Medium | 77% | Medium | ✅ Good |
| Bagged LR | 73.44% | - | - | - | Baseline |
| AdaBoost DTree | 65.58% | - | - | - | ❌ Performance drop |
| Gradient Boost | 91.19% | - | 0% | - | ❌ Imbalanced |
| Simple MLP | ~75% | - | - | - | ✅ Good |
| Complex MLP | ~78% | - | - | - | ✅ Good |

**Final Deployment**: All 6 models (RF, Bagged SVM, Bagged DT, Bagged LR, Simple MLP, Complex MLP) used for ensemble voting

## Chatbot Module

### Core Functionality

**ChatBot Class** (`test_main.py`):
- Intent recognition using Bag-of-Words similarity
- Two primary intents:
  1. **Player Stats Lookup** (intent 0)
  2. **Draft Prediction** (intent 1)
- Multi-model ensemble prediction with probability averaging

### NLP Processing

**User Functions** (`user_functions.py`):
```python
def preprocess(input_sentence):
    # Lowercase, remove punctuation, tokenize, remove stopwords
    input_sentence = input_sentence.lower()
    input_sentence = re.sub(r'[^\w\s]', '', input_sentence)
    tokens = word_tokenize(input_sentence)
    return [i for i in tokens if i not in stop_words]

def compare_overlap(user_message, possible_response):
    # Bag-of-Words similarity scoring
    similar_words = sum(1 for token in user_message 
                       if token in possible_response)
    return similar_words

def extract_nouns(tagged_message):
    # Extract player names via POS tagging
    return [token[0] for token in tagged_message 
            if token[1].startswith("N")]

def compute_similarity(tokens, category):
    # Word2Vec semantic similarity using spaCy
    return [[token.text, category.text, token.similarity(category)] 
            for token in tokens]
```

### Prediction Output

```python
def get_predictions(name, G, A, GP, PIM, position_D, position_LW):
    # Load all models
    rf_model = joblib.load('random_forest_model.joblib')
    svm_model = joblib.load('bag_svm_model.joblib')
    dtree_model = joblib.load('bag_dtree_model.joblib')
    lr_model = joblib.load('bag_lr_model.joblib')
    simple_mlp = tf.keras.models.load_model('simple_MLP_model.h5')
    complex_mlp = tf.keras.models.load_model('complex_MLP_model.h5')
    
    # Scale input
    scaler = joblib.load('scaler.joblib')
    user_data = scaler.transform([[G, A, GP, PIM, position_D, position_LW]])
    
    # Get predictions from all models
    predictions = [model.predict_proba(user_data)[0][1] * 100 
                   for model in models]
    
    # Return ensemble average
    avg_prob = np.mean(predictions).round(2)
    return {
        'rf_prob': predictions[0],
        'svm_prob': predictions[1],
        'dtree_prob': predictions[2],
        'lr_prob': predictions[3],
        'simple_mlp_prob': predictions[4],
        'complex_mlp_prob': predictions[5],
        'average_prob': avg_prob
    }
```

## Installation & Usage

### Setup
```bash
# Install dependencies
pip install pandas numpy scikit-learn tensorflow keras joblib
pip install spacy nltk matplotlib seaborn

# Download NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
```

### Training Models
```python
# Run the complete ML pipeline
jupyter notebook NHL.ipynb
# Execute all cells to train and save models
```

### Using the Chatbot
```python
from NHL_Chat_Bot_Master.test_main import ChatBot

bot = ChatBot()
response = bot.respond("Can you predict draft round?")
print(response)

# Make prediction
prediction = bot.get_predictions(
    name="John Doe",
    G=45, A=60, GP=68, PIM=20,
    position_D=0, position_LW=1
)
print(prediction)
```

### Loading Models Directly
```python
import joblib
import tensorflow as tf

# Load best model
svm_model = joblib.load('NHL_Chat_Bot_Master/bag_svm_model.joblib')
scaler = joblib.load('NHL_Chat_Bot_Master/scaler.joblib')

# Make prediction
player_stats = [[45, 60, 68, 20, 0, 1]]  # G, A, GP, PIM, pos_D, pos_LW
scaled_stats = scaler.transform(player_stats)
probability = svm_model.predict_proba(scaled_stats)[0][1]
print(f"First-round probability: {probability * 100:.2f}%")
```

## Key Insights

### Model Performance
- **Bagging dramatically improved SVM** (~10% accuracy gain)
- **Ensemble methods** consistently outperformed single models
- **Class imbalance** required careful evaluation (recall > precision)
- **Deep learning** competitive but not superior to ensemble methods

### Feature Importance
- **Goals and Assists** most predictive features
- **Position** had minimal variance contribution (PCA)
- **Games Played** important for context (injury vs. skill)
- **Penalty Minutes** weakly correlated with draft position

### Data Quality
- **Temporal consistency** crucial (pre-1990 data unreliable)
- **Player name disambiguation** challenging in dataset merging
- **Outlier detection** essential for model accuracy
- **Junior league normalization** important for fair comparison

## Technologies Used

- **Machine Learning**: scikit-learn, XGBoost-compatible workflow
- **Deep Learning**: TensorFlow 2.x, Keras
- **NLP**: spaCy (word2vec), NLTK
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib, HDF5

## Future Work

- **XGBoost/LightGBM**: Test gradient boosting frameworks
- **Feature Engineering**: Add more derived statistics (points per game, +/-, shooting %)
- **Temporal Models**: LSTM/GRU for career trajectory prediction
- **Transfer Learning**: Pre-trained language models for chatbot
- **Explainability**: SHAP/LIME for feature contribution analysis
- **Cross-validation**: K-fold CV for more robust evaluation

## Contributors

James Hutchins - Machine Learning Engineer

## Acknowledgments

- **Western AI**: Project framework and guidance
- **Kaggle**: Data sources (NHL Draft data, Elite Prospects stats)
- **scikit-learn**: Comprehensive ML toolkit
- **TensorFlow/Keras**: Deep learning framework

---

*This module provides a production-ready machine learning pipeline for NHL draft prediction, demonstrating advanced feature engineering, ensemble methods, and conversational AI integration.*