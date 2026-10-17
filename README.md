# CSAC 212: Introduction to Artificial Intelligence Projects

A collection of Machine Learning, Deep Learning, and Natural Language Processing (NLP) projects completed for **CSAC 212 (Introduction to Artificial Intelligence)**. This repository explores binary classification across structured tabular data and unstructured social media text using **Scikit-Learn**, **TensorFlow/Keras**, and **NLTK**.

> **Note on Datasets & Documentation:** Raw dataset files (`.csv`) are not committed to this repository. Direct links to the original datasets from the **UCI Machine Learning Repository** and **Kaggle** are provided inside each respective Jupyter Notebook and in the sections below. All methodologies, experimental setups, and analytical findings are documented directly in this README.

---

## Repository Structure

```text
csac212-projects/
├── README.md                                 # Comprehensive project documentation & results
├── PS4_TicTacToe_LogisticRegression.ipynb    # Problem Set 4: Classical ML Classification
├── PS5_AuctionVerification_NeuralNet.ipynb   # Problem Set 5: Deep Learning / ANN Classification
└── FinalProject_RedditDepression_NLP.ipynb   # Final Project: NLP Text Classification & LSTM Comparison
```

---

## 1. Problem Set 4: Tic-Tac-Toe Endgame Classification

### Overview & Dataset
This project explores classical supervised learning using **Logistic Regression** on the **Tic-Tac-Toe Endgame Dataset** from the UCI Machine Learning Repository (ID: 101).
- **Dataset Scope:** Encodes the complete set of 958 possible board configurations at the end of tic-tac-toe games where player `x` is assumed to have played first.
- **Features & Target:** 9 categorical attributes (`A1` through `A9`) representing the 9 board squares (`x` = player x, `o` = player o, `b` = blank). The binary target concept `Class` is `positive` (`1`) when player `x` has achieved one of the 8 possible ways to create a "three-in-a-row" (626 instances), and `negative` (`0`) otherwise (332 instances).
- **Dataset Access:** Fetched programmatically inside the notebook via `ucimlrepo` (`fetch_ucirepo(id=101)`). [UCI Dataset Link](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame)

### Methodology
1. **Exploratory Data Analysis (EDA):** Visualized mark frequencies across all 9 board positions using 3x3 countplot grids, positional frequency heatmaps, class distribution bar charts, and a feature correlation heatmap. Player `x` holds the highest frequency across all squares, particularly in the center square (`A5` with 458 occurrences).
2. **Preprocessing:** Mapped categorical board states into numerical values (`x`: `1`, `o`: `-1`, `b`: `0`) and binary target classes (`positive`: `1`, `negative`: `0`). Standardized the feature matrix using `StandardScaler`.
3. **Model Training:** Split the data into training and testing sets and fitted a Scikit-Learn `LogisticRegression` classifier, alongside an implementation exploring the course's `banknotes0.py` classification template.

### Results & Findings
- Evaluating the Logistic Regression model on the testing set demonstrated strong predictive accuracy in linearly separating winning board configurations for `x` from non-winning states.
- Performance was validated using accuracy, precision, recall, F1-score metrics, and a confusion matrix heatmap to inspect True Positives, True Negatives, False Positives, and False Negatives.

---

## 2. Problem Set 5: Auction Verification with Neural Networks

### Overview & Dataset
This project investigates whether costly verification of complex process models (simultaneous multi-round auctions used for frequency spectra) can be replaced with outcome predictions using an **Artificial Neural Network (ANN)**.
- **Dataset Scope:** The **Auction Verification Dataset** from the UCI Machine Learning Repository contains 2,043 instances of auction transaction verification runs.
- **Features & Target:** Each instance consists of 7 input features: the capacities of 4 bidders (`process.b1.capacity` through `process.b4.capacity`) and verification properties for price (`property.price`), product (`property.product`), and winning bidder (`property.winner`). The binary target `verification.result` indicates whether a transaction is verifiable: `True` (`1`, 262 instances) or `False` (`0`, 1,781 instances).
- **Dataset Access:** Loaded locally in the notebook; original data is available at the [UCI Machine Learning Repository (ID: 713)](https://archive.ics.uci.edu/dataset/713/auction+verification).

### Methodology
1. **Correlation Analysis & Preprocessing:** Generated a correlation heatmap across all variables. Features were largely independent, with the highest correlation (`0.444`) occurring between Bidder 1 and Bidder 3's capacities, followed by `0.286` between Bidder 1's capacity and property price. Data was split into 67% training (1,368 samples) and 33% testing (675 samples) and normalized via `StandardScaler`.
2. **Baseline Architecture:** Built an initial Keras `Sequential` network with two hidden `Dense` layers (8 ReLU units each), a 1-unit Sigmoid output layer, `binary_crossentropy` loss, `sgd` optimizer, batch size of 1, and 10 epochs.
3. **Hyperparameter Fine-Tuning:** To resolve loss oscillation and improve generalization, the architecture was expanded to **128 and 64 ReLU units** in the first and second hidden layers. **Dropout layers (0.5 and 0.4)** were introduced to prevent overfitting, the optimizer was upgraded to **Adam**, and training was adjusted to **50 epochs** with a **batch size of 16** (using a 20% validation split).

### Results & Findings

| Model Configuration | Hidden Layers | Regularization | Optimizer | Epochs / Batch | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline ANN** | 8 -> 8 (ReLU) | None | SGD | 10 / Batch=1 | ~86.50% (Val) | 0.3165 (Val) |
| **Fine-Tuned ANN** | 128 -> 64 (ReLU) | Dropout (0.5, 0.4) | Adam | 50 / Batch=16 | **90.67%** | **0.2631** |

- In the baseline model, accuracy plateaued and oscillated around 87% while validation loss hovered between 30% and 40%.
- Fine-tuning the network with wider layers, Dropout regularization, and the Adam optimizer stabilized convergence and boosted testing accuracy to **90.67%** with a reduced loss of **26.31%**.

---

## 3. Final Project: Flagging Depressive Social Media Texts

### Overview & Dataset
Motivated by research linking social media usage with exposure to triggering or depressive content, this project develops an NLP text classification pipeline that flags input texts as **non-depressive (`0`)** or **depressive (`1`)**.
- **Hybrid Dataset Construction:** Compiled a balanced dataset of **7,800 Reddit posts** (3,900 non-depressive and 3,900 depressive) by combining two Kaggle datasets:
  1. *Depression: Reddit Dataset (Cleaned)* — Extracted 3,900 non-depressive (`0`) instances.
  2. *Reddit dataset: r/depression and r/SuicideWatch* — Extracted 3,900 depressive (`1`) instances directly from mental health subreddits for higher labeling credibility.
- **Dataset Access:** Links to both Kaggle datasets are documented inside the notebook: [Depression Reddit Cleaned](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) and [Reddit r/depression & r/SuicideWatch](https://www.kaggle.com/datasets/xavrig/reddit-dataset-rdepression-and-rsuicidewatch).

### Methodology
1. **NLP Preprocessing & Visualization:** Stripped punctuation, numbers, and special characters using regular expressions, converted text to lowercase, and removed English stopwords via `NLTK` (alongside tokenization and lemmatization). Generated WordClouds to contrast dominant vocabulary: non-depressive posts centered on everyday routines (*"work"*, *"day"*, *"going"*, *"sleep"*, *"today"*), whereas depressive posts centered on internal emotional states (*"feel"*, *"want"*, *"life"*, *"know"*, *"people"*, *"think"*).
2. **Splitting & Vectorization:** Split the corpus 80/20 into **6,240 training** and **1,560 testing** instances (`random_state=42`). Encoded texts into numerical feature vectors using **TF-IDF (`TfidfVectorizer`)**.
3. **Model Optimization (Prioritizing Recall):**
   * **Initial Logistic Regression:** Trained with `C=0.1` and `max_iter=500`. While overall accuracy was 93.59%, recall for depressive texts (Class 1) was only **88%**, missing 12% of depressive posts (89 False Negatives).
   * **Fine-Tuned Logistic Regression:** Because missing depressive content is detrimental to content-flagging goals, inverse regularization strength was increased to `C=15` and class weights were adjusted to `class_weight={0: 1, 1: 2}` to penalize False Negatives more heavily. The trained pipeline and vectorizer were serialized using `joblib`.
4. **Deep Learning Benchmark (LSTM):** To benchmark the classical ML model against deep learning, a Keras `Sequential` neural network was trained on the same dataset using an `Embedding` layer (`input_dim=10000`, `output_dim=64`), an **LSTM (32 units)** layer, and a Sigmoid output layer, compiled with `adam` and `EarlyStopping` callbacks.

### Results & Model Comparison

Evaluated on the unseen testing set of **1,560 posts** (795 Non-Depressive `[0]`, 765 Depressive `[1]`):

| Model | Overall Accuracy | Class 0 Precision / Recall | Class 1 Precision / Recall | Macro F1-Score | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial Logistic Regression** (`C=0.1`) | 93.59% | 0.90 / 0.99 | 0.98 / 0.88 | 0.93 | `TN: 784` \| `FP: 11`<br>`FN: 89` \| `TP: 676` |
| **Fine-Tuned Logistic Regression** (`C=15`, weighted) | **97.24%** | **0.97 / 0.98** | **0.98 / 0.97** | **0.97** | `TN: 777` \| `FP: 18`<br>`FN: 25` \| `TP: 740` |
| **LSTM Neural Network** *(Benchmark)* | **98.59% (~99%)** | **0.98 / 0.99** | **0.99 / 0.98** | **0.99** | `TN: 788` \| `FP: 7`<br>`FN: 15` \| `TP: 750` |

### Key Findings
- **Impact of Class Weighting & Regularization Tuning:** Adjusting `C=15` and weighting Class 1 (`{0: 1, 1: 2}`) in the Logistic Regression model reduced missed depressive texts (False Negatives) from **89 down to 25**, boosting Class 1 recall from **88% to 97%** and overall accuracy to **97.24%**.
- **Classical ML vs. Deep Learning:** While the LSTM neural network achieved the highest overall score (**99% F1-score**, with only 7 False Positives and 15 False Negatives), the fine-tuned TF-IDF + Logistic Regression model proved highly competitive—offering near-equivalent 97% accuracy and F1 performance with significantly lower computational overhead and faster inference for real-time batch or single-string text flagging.

---

## How to Run the Notebooks

1. Install the required Python dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras nltk wordcloud ucimlrepo joblib
   ```
2. For **Problem Set 5** and the **Final Project**, download the datasets using the UCI/Kaggle links provided in the sections above and place the `.csv` files in the root directory. (**Problem Set 4** fetches its dataset automatically via `ucimlrepo`).
3. Launch Jupyter Notebook or open the `.ipynb` files in VS Code / Google Colab:
   ```bash
   jupyter notebook
   ```

---

## Author
- **Jessica Mae T. Lanuzo** — BSCS