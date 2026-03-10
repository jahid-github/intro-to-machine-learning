

## SECTION 1 — Introduction to Machine Learning (Lecture 1)

**Q1. What is Machine Learning? How is it different from traditional software development?**

**Answer:** Machine Learning is the science of getting computers to learn without being explicitly programmed. In traditional software, a programmer writes step-by-step rules. In ML, the computer *learns* those rules from data automatically.[^1]

***

**Q2. List the 3 main reasons why ML has exploded in recent years.**

**Answer:**[^1]

- More data available
- Faster computers (better hardware)
- Better algorithms

***

**Q3. When should you NOT use Machine Learning?**

**Answer:**[^1]

- When there is not enough data or data quality is poor
- When a simple rule works fine (e.g., "If temp > 30°C → turn on AC")
- When it is not cost-effective (ML takes time, money, and maintenance)

***

**Q4. What is Supervised Learning? Give two examples.**

**Answer:** Supervised learning learns from labeled data (input-output pairs) to make predictions on new data. The model learns a function X → Y.[^1]

Examples:

- Email spam detection (input: email text → output: spam/not spam)
- House price prediction (input: size, bedrooms → output: price in €)

***

**Q5. What is Unsupervised Learning? How does it differ from supervised learning?**

**Answer:** Unsupervised learning finds patterns in data that has NO labels. There is no correct answer given. The goal is NOT prediction but to understand the structure of the data. Supervised = produces a **prediction model**. Unsupervised = produces a **data model**.[^1]

***

**Q6. (MCQ) Which of the following is an example of unsupervised learning?**

- A) Predicting house prices
- B) Detecting spam emails
- **C) Customer segmentation into groups ✅**
- D) Predicting student exam scores

***

**Q7. List the 7 steps of the Machine Learning Workflow.**

**Answer:**[^1]

1. Collect data
2. Clean \& prepare data
3. Split data (training/test set)
4. Choose hypothesis/model
5. Train the model
6. Evaluate on test set
7. Improve (features, parameters)

***

**Q8. Define: Features (X), Target (y), Training Set, Test Set, Overfitting.**

**Answer:**[^1]

- **Features (X):** Input variables (e.g., size, age, color)
- **Target/label (y):** Correct answer the model predicts (e.g., price, spam/not spam)
- **Training set:** Data used to teach the model patterns
- **Test set:** Separate data to evaluate model on unseen examples
- **Overfitting:** Model memorizes training data and fails on new data

***

## SECTION 2 — ML Basics \& Python Libraries (Lecture 2 / basics-2)

**Q9. What are the three main Python libraries in ML, and what does each do?**

**Answer:**[^2]


| Library | Role |
| :-- | :-- |
| NumPy | Math engine — works with numbers \& arrays |
| Pandas | Data handling — tables (rows \& columns) |
| scikit-learn | Machine Learning — models, training, evaluation |


***

**Q10. How do you decide: Regression or Classification?**

**Answer:**[^2]

- If output is a **number** → Regression (e.g., predict price, score)
- If output is a **category** → Classification (e.g., predict spam/not spam)

***

**Q11. Write the 3 common ways to load data in sklearn.**

**Answer:**[^2]

```python
# Method 1: sklearn built-in dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Method 2: Manual data
X = [[^1],[^2],[^3],[^4]]   # hours studied
y = [50, 60, 70, 80]   # scores

# Method 3: CSV file
import pandas as pd
data = pd.read_csv("students.csv")
X = data[['hours', 'attendance']]
y = data['score']
```


***

**Q12. Write the standard code for Fit → Predict → Evaluate in sklearn.**

**Answer:**[^2]

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Regression → MSE
print(mean_squared_error(y_test, pred))

# Classification → Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
```


***

## SECTION 3 — Linear Regression, Loss Function \& Gradient Descent (Lecture 3)

**Q13. What is the equation of a linear regression model? Explain each term.**

**Answer:**[^3]

$$
\hat{y} = wx + b
$$

- $w$ = weight (slope)
- $b$ = bias (intercept)
- $\hat{y}$ = predicted value

The model tries to find the best values of $w$ and $b$ that minimize error.

***

**Q14. What is Mean Squared Error (MSE)? Why do we square the errors?**

**Answer:** MSE measures how far predictions are from actual values:[^3]

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

We square because:

- It removes negative signs (no cancellation)
- It penalizes big errors more strongly[^3]

***

**Q15. Explain Gradient Descent using the "hill" analogy.**

**Answer:** Think of each possible model $(w, b)$ as a point on a hill. High error = high on the hill, low error = bottom of the hill. Gradient Descent is the process of walking downhill step by step to minimize error:[^3]

1. Predict
2. Measure error
3. Move slightly downhill (update weights)

The **learning rate** controls how big each step is.[^3]

***

**Q16. Write the weight update rule for gradient descent.**

**Answer:**

<img width="454" height="49" alt="image" src="https://github.com/user-attachments/assets/1fcdb4ef-7383-4824-a5a3-a18647b7dfb5" />

We subtract the slope because we want to move toward the minimum (bottom of the hill).

***

**Q17. (MCQ) What happens if the learning rate is too large?**

- A) Model converges quickly and correctly
- **B) Model may overshoot the minimum and never converge ✅**
- C) Model trains slowly
- D) Model underfits

***

## SECTION 4 — Data Cleansing \& Train/Test Split (Lecture 4)

**Q18. List the 5 common data quality issues and one solution for each.**

**Answer:**


| Issue | Solution |
| :-- | :-- |
| Missing values | `data.dropna()` or `data.fillna(mean)` |
| Duplicates | `data.drop_duplicates()` |
| Incorrect data (e.g., Salary = "abc") | `pd.to_numeric(errors='coerce')` |
| Inconsistent text ("Male","M","male") | `str.lower()`, `.replace({})` |
| Outliers | IQR method or boxplot |


***

**Q19. Write the Python commands for data profiling (first look at data).**

**Answer:**

```python
data.head()          # first 5 rows
data.info()          # data types, missing values
data.describe()      # mean, min, max
data.isna().sum()    # count of missing values
```


***

**Q20. What is the IQR method for detecting outliers? Write the code.**

**Answer:**

```python
Q1 = data['column'].quantile(0.25)
Q3 = data['column'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[(data['column'] < lower) | (data['column'] > upper)]
```


***

**Q21. Why do we split data into train and test sets? What is a typical split?**

**Answer:** If we train and test on the same data, the model memorizes answers — this is called **overfitting**. The typical split is **80% training / 20% testing**.[^4]

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


***

**Q22. What is Stratified Split? When and why do you use it?**

**Answer:** Stratified split ensures that each class is represented fairly in both training and test sets. Use when the dataset is imbalanced (e.g., 90% Not Spam, 10% Spam).[^4]

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```


***

## SECTION 5 — KNN Algorithm (Knn.pdf + Week-5-Lecture-10)

**Q23. What is the KNN algorithm? How does it make predictions?**

**Answer:** K-Nearest Neighbors (KNN) is a supervised, non-parametric, lazy learning algorithm. Steps:[^5]

1. Pick a new data point
2. Measure distance to all training points
3. Select k closest neighbors
4. **Majority vote** for classification, **average** for regression[^5]

***

**Q24. What does "lazy learner" mean in KNN?**

**Answer:** KNN does not build a model during training. It stores the entire dataset and performs all computation only at **prediction time**. This is why it is called a lazy learner.[^5]

***

**Q25. Calculate the Manhattan Distance between Person A = (170, 65) and Person B = (175, 70).**

**Answer:**[^5]

$$
d = |170-175| + |65-70| = 5 + 5 = 10
$$

Manhattan distance is like walking in a city grid — only horizontal/vertical moves, no diagonal cutting.

***

**Q26. Calculate the Euclidean Distance between Person A = (170, 65) and Person B = (175, 70).**

**Answer:**[^5]

$$
d = \sqrt{(170-175)^2 + (65-70)^2} = \sqrt{25+25} = \sqrt{50} \approx 7.07
$$

Euclidean is straight-line (diagonal) distance.

***

**Q27. Why is Feature Scaling crucial for KNN? How does StandardScaler work?**

**Answer:** Without scaling, features with large ranges (e.g., Salary: 0–100,000) dominate distance calculations and make smaller features (e.g., Age: 0–100) irrelevant. `StandardScaler` transforms each feature to have **mean = 0** and **standard deviation = 1**:[^5]

$$
z = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```


***

**Q28. Write complete KNN classification code with StandardScaler for Iris dataset.**

**Answer:**[^5]

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```


***

**Q29. What is Cross-Validation? Explain k-Fold Cross-Validation.**

**Answer:** Cross-validation tests the model multiple times on different portions of the data and averages the results. In **k-Fold CV** (e.g., k=5): the dataset is split into 5 folds. Each fold takes a turn as the test set while the rest are training data. Final accuracy = average of 5 scores → **less bias, more reliable**.[^5]

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("CV Accuracy:", scores.mean())
```


***

## SECTION 6 — Confusion Matrix \& Evaluation Metrics (Lectures 6 \& 8)

**Q30. Define TP, FP, TN, FN with a medical diagnosis example.**

**Answer:**[^6]

- **TP (True Positive):** Model predicted sick → person IS sick ✅
- **FP (False Positive):** Model predicted sick → person is NOT sick ❌
- **TN (True Negative):** Model predicted healthy → person IS healthy ✅
- **FN (False Negative):** Model predicted healthy → person IS sick ❌

***

**Q31. Write the formulas for Accuracy, Precision, Recall, and F1-Score.**

**Answer:**[^6]

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

***

**Q32. (Calculation) Given TP=80, FP=10, TN=90, FN=20. Calculate all four metrics.**

**Answer:**[^6]

- **Accuracy** = (80+90)/(80+90+10+20) = 170/200 = **0.85 (85%)**
- **Precision** = 80/(80+10) = 80/90 = **0.89**
- **Recall** = 80/(80+20) = 80/100 = **0.80**
- **F1** = 2×(0.89×0.80)/(0.89+0.80) = **0.842**

***

**Q33. (MCQ) A model has high recall but low precision. What does this mean?**

- A) The class is perfectly handled
- **B) The class is well detected but the model also includes wrong classes ✅**
- C) The model can't detect the class
- D) Model is overfitting

***

## SECTION 7 — Probability, Bayes Theorem \& Naïve Bayes (Lecture 9)

**Q34. What is Prior Probability and Conditional Probability? Give examples.**

**Answer:**[^7]

- **Prior Probability:** Probability before seeing evidence. E.g., P(pos) = 6/12 = 0.5
- **Conditional Probability:** Probability of an event given another has occurred. E.g., P(pos | thick) = 3/8 = 0.375 (probability of liking a pie given it has thick filling)

***

**Q35. State Bayes' Rule and label each component.**

**Answer:**[^7]

$$
P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}
$$

- **Posterior** P(H|E): Updated probability of hypothesis after seeing evidence
- **Likelihood** P(E|H): Probability of evidence given hypothesis is true
- **Prior** P(H): Probability of hypothesis before evidence
- **Marginal** P(E): Probability of evidence under any circumstance

***

**Q36. What is the Naïve Bayes Classifier and when is it used?**

**Answer:** Naïve Bayes is a probabilistic classifier based on Bayes' Rule. It assumes all features are **independent** (the "naïve" assumption). It is fast, handles noisy data well, and is used in spam detection, medical diagnosis, and robotics (LiDAR-based obstacle detection).[^7]

***

## SECTION 8 — Clustering \& K-Means (Lecture 7)

**Q37. What is clustering? What type of learning is it?**

**Answer:** Clustering is the classification of objects into different groups so that data in each group shares common traits. It is **unsupervised learning** — no labels are given.[^8]

***

**Q38. Explain the K-Means algorithm step by step.**

**Answer:**

1. Choose k initial centroids
2. Assign each data point to the nearest centroid
3. Recalculate centroids as the mean of each cluster
4. Repeat until centroids don't change (convergence)

Convergence = cluster means don't change OR cluster assignment doesn't change.

***

**Q39. What is the Elbow Method for choosing k in K-Means?**

**Answer:** Plot **WCSS (Within-Cluster Sum of Squares)** against different values of k. WCSS decreases as k increases. The **"elbow"** point — where the improvement slows down significantly — is the optimal k.[^8]

***

**Q40. What is the difference between Hierarchical and Partitional Clustering?**

**Answer:**[^8]


| Type | Description |
| :-- | :-- |
| Partitional (K-Means) | Partitions are independent; k must be specified |
| Hierarchical | Uses a tree structure (dendrogram); k does NOT need to be pre-specified |


***

**Q41. Name 4 distance metrics and when to use each.**

**Answer:**[^8]


| Metric | Use Case |
| :-- | :-- |
| **Euclidean** | Small to medium dimensions; straight-line distance |
| **Manhattan** | High-dimensional data; discrete/binary attributes |
| **Cosine** | Text/recommendation systems where direction matters more than magnitude |
| **Hamming** | Binary strings; error detection in networks |


***

## SECTION 9 — Feature Selection, PCA \& Dimensionality Reduction (Lecture 8)

**Q42. What is Feature Selection and why is it important?**

**Answer:** Feature selection is the process of choosing only the most useful input features for an ML model. Benefits:[^9]

- Removes irrelevant/redundant features
- Reduces overfitting
- Increases model training speed
- Makes models simpler to interpret[^9]

***

**Q43. What is the difference between Feature Selection and Feature Extraction?**

**Answer:**

- **Feature Selection:** Selects a subset of *original* features; no new features created
- **Feature Extraction (e.g., PCA):** Creates *new* features (principal components) from combinations of original ones[^9]

***

**Q44. What is PCA (Principal Component Analysis)? List its 6 steps.**

**Answer:** PCA reduces dimensionality by creating new features (principal components) that capture maximum variance. It is an **unsupervised** method introduced by Karl Pearson in 1901.[^9]

Steps:

1. Data centering (subtract mean)
2. Compute covariance matrix
3. Compute eigenvalues
4. Compute eigenvectors
5. Order eigenvectors by eigenvalue
6. Compute principal component scores[^9]

***

**Q45. What is Covariance vs Correlation?**

**Answer:**[^9]

- **Covariance:** Measures the *direction* of relationship between two variables (positive/negative/zero). Does NOT measure strength.
- **Correlation (Pearson r):** Measures the *strength and direction* of a linear relationship. It is dimensionless (−1 to +1).[^9]

***

**Q46. List the 4 types of Feature Scalers with their function.**

**Answer:**[^9]


| Scaler | Function |
| :-- | :-- |
| **StandardScaler** | Mean = 0, Std = 1 |
| **MinMaxScaler** | Scales values between 0 and 1 |
| **MaxAbsScaler** | Divides by absolute max value of each column |
| **RobustScaler** | Robust to outliers using IQR |


***

## SECTION 10 — Neural Networks \& Deep Learning (Week-5 / Lecture 10)

**Q47. How is a biological neuron modeled mathematically?**

**Answer:**[^10]


| Biological Part | Math Model |
| :-- | :-- |
| Dendrites | Inputs (x) and Weights (w) |
| Soma | Weighted sum: $z = \sum w_i x_i$ |
| Axon (action potential) | Activation function |
| Output | Prediction |

$$
z = w_0x_0 + w_1x_1 + \dots + w_Nx_N
$$

***

**Q48. What is the difference between Machine Learning and Deep Learning?**

**Answer:**[^10]

- **ML:** Feature extraction is a **manual** process, followed by learning
- **Deep Learning:** Feature extraction and learning are **integrated** into a single process using multiple neural network layers[^10]

***

**Q49. What are derivatives and why are they important for gradient descent in neural networks?**

**Answer:** The derivative measures the rate of change of a function at a point. In gradient descent, the derivative tells us which direction to adjust weights:[^9]

- **Positive derivative:** function is increasing → move in negative direction
- **Negative derivative:** function is decreasing → move in positive direction[^9]

A derivative of zero indicates a stationary point (minimum, maximum, or saddle point).[^9]

***

**Q50. What is the CRISP-DM framework?**

**Answer:** CRISP-DM (Cross Industry Standard Process for Data Mining) is a 6-phase data science process:[^7]

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (Visualization \& Presentation)[^7]

***

## QUICK-FIRE MCQ ROUND

**Q51.** Which Python library is used for training ML models?
→ **scikit-learn**[^2]

**Q52.** What does `data.dropna()` do?
→ **Removes rows with missing values**[^4]

**Q53.** Default distance metric in KNeighborsClassifier?
→ **Euclidean (Minkowski with p=2)**[^5]

**Q54.** What does `stratify=y` do in `train_test_split`?
→ **Preserves class distribution in both train and test sets**[^4]

**Q55.** In k-Fold CV with k=5, how many times is the model trained?
→ **5 times**[^5]

**Q56.** An F1-score of 0.7 or higher is generally considered?
→ **Good**[^6]

**Q57.** PCA is what type of learning?
→ **Unsupervised**[^9]

**Q58.** What is a dendrogram used for?
→ **Visualizing hierarchical clustering**[^8]

**Q59.** What is the "naive" assumption in Naïve Bayes?
→ **All features are independent of each other**[^7]

**Q60.** Saddle points in deep learning are points where:
→ **Derivative is zero but it is neither a minimum nor maximum**[^9]

***
