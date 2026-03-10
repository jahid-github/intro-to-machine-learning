

### Section 1: ML Fundamentals & Workflow

**1. Short Answer:** What is the CRISP-DM framework, and what are its six sequential phases?


**Answer:** CRISP-DM stands for Cross Industry Standard Process for Data Mining. It is a process model that serves as the base for a data science process. The six sequential phases are: Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, and Deployment.

**2. MCQ:** Which Python library is primarily used as the math engine for handling arrays and fast mathematical operations in machine learning?
A) Pandas
B) NumPy
C) Scikit-learn
D) Matplotlib


**Answer: B.** NumPy works with numbers and arrays, and it is used when data is purely numeric and fast math is required. Pandas handles tabular data, while scikit-learn is used for the actual machine learning models.

**3. Short Answer:** What is the difference between an Epoch, Batch Size, and an Iteration in neural network training?


**Answer:** * **Batch Size:** The total number of training examples present in a single batch when the dataset is too large to feed into the computer all at once.

* 
**Iterations:** The number of batches needed to complete one full epoch.


* 
**Epoch:** One entire passing of all the training data through the algorithm, comprising one forward pass and one backward pass.



### Section 2: Data Cleansing & Preparation

**4. Short Answer:** Explain the difference between Feature Selection and Feature Extraction.


**Answer:** Feature Selection is the process of choosing only the most useful specific features from the original list for a data sample, discarding the rest without generating new features. Feature Extraction methods, such as PCA, extract and generate *new* features from the original list, meaning the reduced subset contains features that were not in the original set.

**5. Coding Example: Identifying Outliers & Stratified Splitting**
Before splitting data, it is crucial to handle outliers and balance the classes. The IQR method removes extreme values falling outside $(Q1 - 1.5 * IQR)$ or $(Q3 + 1.5 * IQR)$. When splitting, using stratification ensures the train and test sets have the same class proportions as the original dataset.

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Visualizing outliers using seaborn (as recommended in lectures)
# sns.boxplot(x=data['columnname']) 

# 2. Stratified Train/Test Split
# Assuming X is your feature matrix and y is your target labels
# test_size=0.2 means 20% of the data goes to the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

```

### Section 3: Regression & Optimization

**6. Short Answer:** In the simple linear regression function $\hat{y} = wx + b$, what do the variables represent?


**Answer:** $\hat{y}$ is the predicted value, $w$ is the weight or slope, $x$ represents the input features, and $b$ is the bias or intercept.

**7. Short Answer:** What role does the learning rate play in Gradient Descent?


**Answer:** Gradient descent repeatedly adjusts model parameters to reduce prediction error. The model takes small steps "downhill" to minimize the error, and the learning rate controls exactly how big each of these steps is.

### Section 4: Distance Metrics & K-Nearest Neighbors (KNN)

**8. MCQ:** Which distance metric is based on an L1-norm, calculates distance by moving at right angles, and is often called the "Taxicab" distance?
A) Euclidean Distance
B) Manhattan Distance
C) Cosine Distance
D) Hamming Distance


**Answer: B.** The Manhattan distance is based on an L1-norm and calculates distance by moving at right angles, making it the Taxicab or City-Block distance.

**9. Short Answer:** Why must data be scaled before applying an algorithm like KNN?
**Answer:** Distance algorithms assume all features are equally important. Raw data violates this assumption, meaning features with larger scales may dominate the results. Scaling (like Standardization) removes unit differences and makes all features comparable.

**10. Coding Example: Feature Scaling**

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standard Scaler scales values such that the mean is 0 and standard deviation is 1
scaler = StandardScaler()

# Example training data
X_train = np.array([[25, 50000], [30, 60000], [35, 70000]])

# Fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

```

### Section 5: Artificial Neural Networks (ANN)

**11. Short Answer:** Compare a Biological Neural Network (BNN) to an Artificial Neural Network (ANN). What do the "Soma" and "Synapse" correspond to in an ANN?


**Answer:** In an ANN, the "Soma" corresponds to the Node, and the "Synapse" corresponds to the Weights or Interconnections. While BNNs are massively parallel and slow but superior in tolerating ambiguity, ANNs are fast but inferior to BNNs and require highly structured data to tolerate ambiguity.

### Section 6: Evaluation Metrics & Probabilities

**12. Short Answer:** Define the F1-score and explain when it is most useful.


**Answer:** The F1-score is a weighted average of true positive (recall) and precision, calculated as $2 * (Precision * Recall) / (Precision + Recall)$. It is particularly suitable when recall and precision must be optimized simultaneously, especially when dealing with imbalanced datasets.

**13. MCQ:** What does a "False Negative" represent in a confusion matrix?
A) The model predicted true and it is true.
B) The model predicted false and it is true.
C) The model predicted false and it is false.
D) The model predicted true and it is false.


**Answer: B.** A False Negative (FN) means the model predicted false, but the actual true label is true (e.g., predicting someone is not sick when they actually are).

### Section 7: Unsupervised Learning (Clustering & PCA)

**14. Short Answer:** How does hierarchical clustering differ from partitional clustering (like K-Means), and how is it visualized?
**Answer:** Partitional clustering creates partitions that are independent of each other. In contrast, hierarchical clustering gives a clustering at multiple levels of granularity and does not require the number of clusters (K) to be specified upfront. It is visualized using a tree structure known as a dendrogram.

**15. Short Answer:** Outline the steps required to compute Principal Component Analysis (PCA).
**Answer:** The steps to compute PCA are:

1. Data centering
2. Compute the covariance matrix
3. Compute the eigenvalue of the covariance matrix
4. Compute the eigenvector of the covariance matrix
5. Order the eigenvector
6. Compute the principal components.
