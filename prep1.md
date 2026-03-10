# Intro to Machine Learning
---
## Part A: Very Important Short Questions with Answers

### 1. What is machine learning?

**Answer:**
Machine learning is letting the computer learn patterns from data and use those patterns to make predictions or decisions. 

### 2. What are X and y in machine learning?

**Answer:**

* **X** = input features, the information used for prediction
* **y** = target/output, what we want to predict
  Example: if we predict exam score from hours studied and attendance, then:
* X = hours studied, attendance
* y = exam score 

### 3. How do you identify whether a problem is regression or classification?

**Answer:**

* If output is a **number**, it is **regression**
* If output is a **category/class**, it is **classification**
  Example:
* Predict house price → regression
* Predict spam/not spam → classification 

### 4. Which baseline model should you start with?

**Answer:**

* Regression → start with **Linear Regression**
* Classification → start with **KNN** or **Logistic Regression** 

### 5. What is the standard ML workflow?

**Answer:**
Read the question → identify y → choose X → choose model → fit → predict → evaluate. 

### 6. What is the role of NumPy, Pandas, and scikit-learn?

**Answer:**

* **NumPy**: arrays, vectors, numerical math
* **Pandas**: tables, CSV/Excel data handling, cleaning
* **scikit-learn**: ML models, train/test split, metrics, training and prediction 

### 7. What is a feature?

**Answer:**
A feature is a measurable property or characteristic used by the model for learning. 

### 8. What is a dataset?

**Answer:**
A dataset is a collection of samples/examples used for machine learning. 

### 9. What is an algorithm in ML?

**Answer:**
An algorithm is a procedure run on data to create a machine learning model. 

### 10. Name the main types of machine learning.

**Answer:**

* Supervised learning
* Unsupervised learning
* Reinforcement learning
* Semi-supervised learning 

### 11. What is supervised learning?

**Answer:**
Supervised learning learns from labeled data, where inputs and correct outputs are given. Example: classification and regression. 

### 12. What is unsupervised learning?

**Answer:**
Unsupervised learning works with unlabeled data and tries to find patterns such as clusters. 

### 13. What is reinforcement learning?

**Answer:**
Reinforcement learning learns by interacting with an environment and receiving rewards or punishments. The PacMan example in the lecture is a reinforcement learning example. 

---

## Part B: Dataset, Split, Validation

### 14. What is the training set?

**Answer:**
The training set is the data the model sees and learns from. 

### 15. What is the test set?

**Answer:**
The test set is only used to evaluate model performance on unseen data. It must not overlap with training data. 

### 16. What is the validation set?

**Answer:**
The validation set is used during development to tune hyperparameters and monitor performance to avoid overfitting. Often around 10–20% of the training data is used for validation. 

### 17. Why do we need train/test split?

**Answer:**
If we train and test on the same data, the model may memorize the answers and show misleadingly high accuracy. This is overfitting. 

### 18. What is a common train/test ratio?

**Answer:**
A typical split is **80% training** and **20% testing**. 

### 19. What does `random_state=42` do?

**Answer:**
It makes the split reproducible, so the same random split can be obtained again. This helps in consistent results.

### 20. What is stratified split?

**Answer:**
Stratified split keeps the same class proportions in train and test sets as in the original dataset. This is especially important for imbalanced classification problems. 

### 21. Give the Python syntax for train/test split.

**Answer:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

For stratified split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```



### 22. What problem does cross-validation solve?

**Answer:**
A single train/test split may be lucky or unlucky. Cross-validation tests the model multiple times on different splits and averages the results, giving a more reliable estimate of performance. 

### 23. What is 5-fold cross-validation?

**Answer:**
The data is divided into 5 equal parts. The model is trained 5 times. Each time, one fold is used as test data and the remaining 4 folds are used for training. Then the 5 scores are averaged. 

---

## Part C: Data Cleansing

### 24. Why does data cleansing matter?

**Answer:**
Real-world data is messy. Bad data leads to wrong insights, and machine learning models depend strongly on data quality. 

### 25. Name common data quality issues.

**Answer:**

* Missing values
* Duplicates
* Incorrect data
* Inconsistent data
* Outliers 

### 26. What is data profiling?

**Answer:**
Before cleaning, inspect the dataset using:

* `data.head()`
* `data.info()`
* `data.describe()`
* `data.isna().sum()`
  This helps understand structure, datatypes, and missing values. 

### 27. How do you handle missing values?

**Answer:**
Two common ways:

* Drop rows/columns if only few values are missing
* Fill values (imputation)

  * numeric → mean or median
  * text → mode 

### 28. How do you fix wrong data types?

**Answer:**

* Convert strings to numbers using `to_numeric()`
* Convert to date using `to_datetime()`
* Often use `errors='coerce'` to handle invalid values 

### 29. How do you make text consistent?

**Answer:**
Use:

```python
data['col'] = data['col'].str.lower()
data['col'] = data['col'].replace({...})
```



### 30. How do you remove duplicates?

**Answer:**
Use:

```python
data.duplicated()
data.drop_duplicates()
```



### 31. What is an outlier?

**Answer:**
An outlier is a value very different from most other values in the dataset. Example: salary = 1,000,000 while others are around 3,000. 

### 32. Should every outlier be removed?

**Answer:**
No. Not every outlier is wrong. It depends on context. For example, a CEO salary may genuinely be much higher. 

### 33. What is the IQR method for outliers?

**Answer:**

* Compute Q1 = 25th percentile
* Compute Q3 = 75th percentile
* IQR = Q3 − Q1
  Values below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR` are often treated as outliers. 

---

## Part D: Linear Regression, Loss, Gradient Descent

### 34. What is linear regression?

<img width="994" height="211" alt="image" src="https://github.com/user-attachments/assets/b764a995-3209-4612-95e6-ff4a8424af0c" />
<img width="295" height="199" alt="image" src="https://github.com/user-attachments/assets/e51bcce7-9533-465f-9d40-4dbffa54ee1b" />

### 35. What does the model learn in linear regression?

**Answer:**
It learns the best values of **w** and **b** so that prediction error becomes as small as possible. 

### 36. What is prediction error?

**Answer:**
<img width="226" height="40" alt="image" src="https://github.com/user-attachments/assets/99eadd28-eb5c-4ba8-8974-5028c7cb33b0" />

It is the difference between actual value and predicted value. 

### 37. What is MSE?

**Answer:**
MSE stands for Mean Squared Error:
<img width="372" height="100" alt="image" src="https://github.com/user-attachments/assets/4a95a77f-5b21-4819-a13b-18d389e8bd31" />

It is used to measure regression error. Smaller MSE means better predictions. 

### 38. Why do we square the error in MSE?

**Answer:**

* Removes negative signs
* Penalizes large errors more strongly 

### 39. What is gradient descent?

**Answer:**
Gradient descent is an optimization method that repeatedly updates model parameters to reduce loss. It moves “downhill” on the loss curve toward minimum error.  

### 40. What is learning rate?

**Answer:**
Learning rate controls the step size during gradient descent.

* Too large → may overshoot
* Too small → learning becomes slow 

### 41. Give the gradient descent update rule.

**Answer:**
<img width="351" height="79" alt="image" src="https://github.com/user-attachments/assets/c7bc36a4-2f91-4857-ab54-770679254079" />

<img width="346" height="43" alt="image" src="https://github.com/user-attachments/assets/034d7ed5-9790-48c8-91e8-a67d6fc4b8ef" />
 

### 42. Why do we subtract the gradient?

**Answer:**
Because the gradient points toward increasing loss. Subtracting it moves the model toward lower loss. 

### 43. What is a derivative?

**Answer:**
A derivative gives the rate of change of a function at a point. It tells how the function changes when the input changes slightly. 

### 44. What happens when derivative is zero?

**Answer:**
The function is flat at that point. It may be a maximum, minimum, or saddle point. 

### 45. How does the second derivative help?

**Answer:**

* (f'(x)=0) and (f''(x)>0) → minimum
* (f'(x)=0) and (f''(x)<0) → maximum
* (f'(x)=0) and (f''(x)=0) → may be saddle point 

### 46. What is a saddle point?

**Answer:**
A saddle point is a point where derivative is zero, but it is neither a maximum nor a minimum. It is common in deep learning loss functions. 

### 47. What are epoch, batch size, and iteration?

**Answer:**

* **Epoch**: one full pass through the entire training dataset
* **Batch size**: number of training examples in one batch
* **Iteration**: number of batches processed to complete one epoch 

---

## Part E: KNN

### 48. What is KNN?

**Answer:**
K-Nearest Neighbors is a supervised ML algorithm used mainly for classification, but it can also be used for regression. It predicts by looking at the nearest data points. 

### 49. Why is KNN called non-parametric?

**Answer:**
Because it does not assume a fixed model form like a line or curve. It stores the data and uses distances directly. 

### 50. Why is KNN called a lazy learner?

**Answer:**
Because it does not build a model during training. It waits until prediction time and then computes distances. 

### 51. Write the basic KNN steps.

**Answer:**

1. Take a new data point
2. Measure distance to all training points
3. Select the k nearest neighbors
4. Use majority vote for classification or average for regression 

### 52. What is the Euclidean distance formula?

**Answer:**
<img width="369" height="147" alt="image" src="https://github.com/user-attachments/assets/2e912a25-726f-4d2e-b0ce-e2a79fd41db7" />

It is straight-line distance. 

### 53. What is Manhattan distance?

**Answer:**
<img width="277" height="109" alt="image" src="https://github.com/user-attachments/assets/e2009197-017c-40ab-8626-28118f191ca6" />

It adds absolute differences feature by feature. It is also called taxicab or city-block distance.  

### 54. When does Euclidean work well?

**Answer:**
Euclidean distance generally works well in low to medium dimensional spaces. 

### 55. When may Manhattan distance be better?

**Answer:**
Manhattan often works better in higher dimensions and when movement along axes is more realistic.  

### 56. Why is feature scaling important in KNN?

**Answer:**
If one feature has a much larger range than another, it dominates the distance calculation and makes distance unfair. 

### 57. What does StandardScaler do?

**Answer:**
It transforms each feature so that:

* mean becomes 0
* standard deviation becomes 1
  This makes features comparable. 

### 58. Does StandardScaler force values into 0 to 1?

**Answer:**
No. StandardScaler does not produce a fixed range. Values are centered around 0 with standard deviation 1. Many values lie roughly between -2 and +2, but outliers can go beyond that. 

### 59. How does k affect KNN?

**Answer:**

* Small k → sensitive to noise, may overfit
* Large k → smoother, but may underfit and ignore local patterns
  So k should be chosen carefully. 

### 60. State some advantages of KNN.

**Answer:**

* Simple to understand
* No training step
* Few parameters
* Works for classification and regression 

### 61. State some disadvantages of KNN.

**Answer:**

* Slow on large datasets
* Struggles with many features
* Can overfit when data is high-dimensional or noisy 

### 62. Mention applications of KNN.

**Answer:**

* Recommendation systems
* Spam detection
* Customer segmentation
* Speech recognition 

---

## Part F: Clustering and Distance Measures

### 63. What is clustering?

**Answer:**
Clustering is grouping similar data points into subsets called clusters based on similarity or distance. 

### 64. Is clustering supervised or unsupervised?

**Answer:**
Clustering is an **unsupervised learning** task because it works without labels. 

### 65. Give some applications of clustering.

**Answer:**

* Document/image/webpage clustering
* Image segmentation
* Web-search result grouping
* Social network grouping
* Gene expression data analysis 

### 66. What are the main types of clustering in the lecture?

**Answer:**

* Flat/Partitional clustering
* Hierarchical clustering 

### 67. What is hierarchical clustering?

**Answer:**
It starts by treating each point as its own cluster and then merges clusters step by step based on distance. Results are visualized using a dendrogram. 

### 68. What is a dendrogram?

**Answer:**
A dendrogram is a tree-like diagram showing the hierarchical relationships between clusters. 

### 69. What is single linkage?

**Answer:**
In single linkage, the distance between two clusters is the distance between their **closest** members. We merge the two clusters with the smallest such distance. 

### 70. What is complete linkage?

**Answer:**
In complete linkage, the distance between two clusters is the **largest** pairwise distance between points in the clusters. We merge the clusters with the smallest of these maximum distances. 

### 71. What is K-means?

**Answer:**
K-means is a flat clustering algorithm that groups (n) objects into (k) clusters based on distance to cluster centroids. 

### 72. What are the basic K-means steps?

**Answer:**

1. Choose number of clusters (k)
2. Initialize centroids
3. Compute distances from points to centroids
4. Assign each point to nearest centroid
5. Recompute centroids
6. Repeat until convergence 

### 73. When does K-means converge?

**Answer:**
K-means can be considered converged when:

* cluster means do not change
* cluster loss does not change much
* cluster assignments do not change 

### 74. What is cosine similarity/distance used for?

**Answer:**
It measures orientation between vectors and is useful in high-dimensional data where magnitude matters less, such as text analysis and recommendation systems. 

### 75. What is Hamming distance?

**Answer:**
Hamming distance measures dissimilarity between equal-length binary vectors or strings by counting differing positions. 

---

## Part G: Confusion Matrix and Metrics

### 76. What is a confusion matrix?

**Answer:**
A confusion matrix summarizes classification results using:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN) 

### 77. Give the formula for accuracy.

**Answer:**
<img width="493" height="94" alt="image" src="https://github.com/user-attachments/assets/c00410be-d637-4e00-a1e0-fdb89868db1f" />


### 78. Give the formula for precision.

**Answer:**
<img width="319" height="73" alt="image" src="https://github.com/user-attachments/assets/29c9983e-4127-494d-9f60-c6775dd13796" />


### 79. Give the formula for recall.

**Answer:**
<img width="289" height="94" alt="image" src="https://github.com/user-attachments/assets/438be0e0-b444-41d0-ad31-612b5a8fc172" />


### 80. Give the formula for F1-score.

**Answer:**
<img width="412" height="78" alt="image" src="https://github.com/user-attachments/assets/945dbeb8-54e8-47cb-85ad-fc4841d3970b" />


### 81. Using TN=55, FP=5, FN=10, TP=30, calculate the metrics.

**Answer:**
Total = 55 + 5 + 10 + 30 = 100

<img width="667" height="355" alt="image" src="https://github.com/user-attachments/assets/f8e23cff-d01e-4369-aebb-b8b20c2ccbfd" />


---

## Part H: Feature Selection, Covariance, Correlation, Scaling, PCA

### 82. What is feature selection?

**Answer:**
Feature selection means choosing only the most useful input features for a model. It removes irrelevant or redundant features, reduces overfitting, speeds training, and makes the model easier to interpret. 

### 83. What is covariance?

**Answer:**
Covariance measures the direction of joint variation between two variables. It tells whether they increase/decrease together, but not the strength clearly. 

### 84. What does covariance sign mean?

**Answer:**

* Cov(x, y) > 0 → x and y increase together
* Cov(x, y) < 0 → one increases while the other decreases
* Cov(x, y) ≈ 0 → little or no linear relation 

### 85. What is correlation?

**Answer:**
Correlation measures the strength of relationship between variables and is dimensionless. 

### 86. What is Pearson correlation coefficient?

**Answer:**
It is the linear correlation coefficient (R), which measures degree of linear relationship between two variables. Values close to +1 show strong positive relation, close to -1 show strong negative relation, and near 0 show weak linear relation. 

### 87. Why is feature scaling needed?

**Answer:**
Features may have different scales. Large-scale features can dominate distance-based comparisons and optimization. Scaling makes features comparable and helps stabilize learning algorithms. 

### 88. Name feature scaling methods from the lecture.

**Answer:**

* MinMaxScaler
* StandardScaler
* MaxAbsScaler
* RobustScaler
* Quantile Transformer
* Log Transformation 

### 89. What does MinMaxScaler do?

**Answer:**
It scales feature values between 0 and 1. 

### 90. What does StandardScaler do?

**Answer:**
It transforms values so the feature has mean 0 and standard deviation 1. 

### 91. What is PCA?

**Answer:**
Principal Component Analysis is an unsupervised dimensionality reduction technique. It creates new features called principal components that capture maximum variance while reducing dimensionality.  

### 92. Is PCA feature selection or feature extraction?

**Answer:**
PCA is **feature extraction**, because it creates new features rather than selecting original ones directly. 

### 93. Why is PCA useful?

**Answer:**

* Reduces dimensionality
* Speeds up training
* Reduces computation
* Helps visualization
* Reduces noise 

### 94. What are PCA steps?

**Answer:**

1. Data centering
2. Compute covariance matrix
3. Compute eigenvalues
4. Compute eigenvectors
5. Order eigenvectors
6. Compute principal components 

---

## Part I: MCQs with Answers

### 95. If the target is “house price,” the problem type is:

A. Classification
B. Regression
C. Clustering
D. Reinforcement learning
**Answer:** B

### 96. Which library is mainly used for tables and CSV files?

A. NumPy
B. Pandas
C. Matplotlib
D. TensorFlow
**Answer:** B 

### 97. Which algorithm is lazy learning?

A. Linear Regression
B. KNN
C. K-means
D. PCA
**Answer:** B 

### 98. Which metric is commonly used for regression?

A. Accuracy
B. F1-score
C. MSE
D. Recall
**Answer:** C 

### 99. Which metric is commonly used for classification?

A. Accuracy
B. MSE
C. RMSE
D. Variance
**Answer:** A 

### 100. Which split preserves class balance?

A. Random split
B. Stratified split
C. Sequential split
D. Manual split
**Answer:** B 

### 101. Which distance is straight-line distance?

A. Manhattan
B. Hamming
C. Euclidean
D. Cosine
**Answer:** C 

### 102. Which distance is used for binary strings?

A. Euclidean
B. Hamming
C. Cosine
D. Pearson
**Answer:** B 

### 103. In KNN, if k becomes too large, the model may:

A. Overfit more
B. Underfit
C. Become unsupervised
D. Become linear regression
**Answer:** B 

### 104. Which is unsupervised?

A. Linear Regression
B. KNN
C. PCA
D. Logistic Regression
**Answer:** C 

### 105. In gradient descent, learning rate controls:

A. Number of features
B. Step size
C. Number of classes
D. Number of labels
**Answer:** B 

---

## Part J: Coding Questions with Answers

### 106. Write Python code for a simple linear regression example.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# sample data
data = pd.DataFrame({
    'hours': [1, 2, 3, 4, 5, 6],
    'score': [50, 55, 65, 70, 80, 90]
})

X = data[['hours']]
y = data['score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

print("Predictions:", pred)
print("MSE:", mse)
```

### 107. Write Python code for KNN classification with scaling.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)
```

This matches the lecture ideas: use train/test split, scaling, KNN, and accuracy for classification.  

### 108. Write code to inspect missing values and duplicates.

```python
import pandas as pd

data = pd.read_csv("data.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data.isna().sum())
print(data.duplicated().sum())
```



### 109. Write code to fill missing numeric values with mean.

```python
data['age'] = data['age'].fillna(data['age'].mean())
```

### 110. Write code to remove duplicates.

```python
data = data.drop_duplicates()
```

### 111. Write code for confusion matrix metrics.

```python
tp = 30
tn = 55
fp = 5
fn = 10

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```

### 112. Write a code skeleton for K-means clustering.

```python
from sklearn.cluster import KMeans
import pandas as pd

data = pd.DataFrame({
    'x': [2, 2, 8, 5, 7, 6, 1, 4],
    'y': [10, 5, 4, 8, 5, 4, 2, 9]
})

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

print("Cluster labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)
```

---

## Part K: Likely Exam Theory Questions

### 113. Explain overfitting and underfitting.

**Answer:**

* **Underfitting**: model is too simple, fails to capture important patterns
* **Overfitting**: model memorizes training data too closely and performs badly on new data
  The lecture visually shows underfitted, good fit, and overfitted models. 

### 114. Explain why scaling helps gradient descent.

**Answer:**
When features have very different scales, optimization becomes unstable and slow. Scaling makes the loss surface easier to optimize and helps gradient descent move more smoothly. 

### 115. Explain why distance choice matters in ML.

**Answer:**
Algorithms like KNN and clustering rely directly on distance. If the distance measure is not suitable, “similarity” is measured wrongly, so predictions or clusters become poor.  

### 116. Explain the difference between feature selection and feature extraction.

**Answer:**

* **Feature selection** keeps some original features and discards others
* **Feature extraction** creates new features from the old ones
  PCA is feature extraction. 

### 117. Explain CRISP-DM briefly.

**Answer:**
CRISP-DM is a data science process model with six phases:

1. Business understanding
2. Data understanding
3. Data preparation
4. Modeling
5. Evaluation
6. Deployment 

---

## Part L: 10 Rapid-Fire Revision Answers

### 118. Number output or category output?

**Answer:** Number → regression, category → classification. 

### 119. KNN training time?

**Answer:** Almost no true training; most work happens at prediction time. 

### 120. Best first check before cleaning data?

**Answer:** Data profiling with `head()`, `info()`, `describe()`, `isna().sum()`. 

### 121. Why not test on training data?

**Answer:** Because it gives misleading high performance and causes overfitting. 

### 122. MSE is for?

**Answer:** Regression. 

### 123. Accuracy is for?

**Answer:** Classification. 

### 124. StandardScaler output mean and std?

**Answer:** Mean 0, standard deviation 1. 

### 125. Hierarchical clustering output visualization?

**Answer:** Dendrogram. 

### 126. Cosine similarity is good for?

**Answer:** High-dimensional data like text or recommendation systems. 

### 127. PCA is supervised or unsupervised?

**Answer:** Unsupervised. 

---

What is KNN?

Answer:
KNN is a supervised machine learning algorithm that predicts a new sample using the nearest training samples.
For classification, it uses majority vote. For regression, it uses averaging.
It depends on distance measures like Euclidean or Manhattan distance.
Feature scaling is important because large-scale features can dominate distance.

Good formula-answer style

Example: What is accuracy?

Answer:
Accuracy measures the proportion of correct predictions out of all predictions.

Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Good theory-answer style

Example: Why is scaling important?

Answer:
Scaling is important because machine learning features may have very different ranges.
Distance-based algorithms like KNN and clustering are affected strongly by feature magnitude.
Scaling makes features comparable and improves learning stability.

Final 20 Must-Memorize Points

X = features, y = target

Number output -> regression

Category output -> classification

Supervised learning uses labeled data

Unsupervised learning uses unlabeled data

Train set learns, test set evaluates

Validation set helps tune model

Overfitting = memorizing training data

Underfitting = model too simple

MSE is for regression

Accuracy/Precision/Recall/F1 are for classification

KNN is lazy learning

KNN needs scaling

Euclidean = straight-line distance

Manhattan = sum of absolute differences

Clustering is unsupervised

K-means uses centroids

StandardScaler -> mean 0, std 1

PCA reduces dimensionality

PCA is feature extraction, not feature selection
