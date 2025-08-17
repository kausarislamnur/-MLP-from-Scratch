# MLP from Scratch: Solving the XOR Classification Task

## 📌 Project Overview
This project implements a **2-2-1 Multilayer Perceptron (MLP)** from scratch in **NumPy** to solve the classic **XOR classification problem**.  
The network is trained using **backpropagation** with **Mean Squared Error (MSE) loss** and manual weight updates (no deep learning frameworks).  

We also evaluate the model using common classification metrics and visualize its performance with a **loss curve** and **ROC curve**.  

---

## 🎯 Objectives
- Build a **2-2-1 MLP**:
  - **2 input neurons** (for XOR inputs)  
  - **2 hidden neurons** (activation: Sigmoid or ReLU)  
  - **1 output neuron** (Sigmoid activation)  
- Train using **backpropagation** and **MSE loss**.  
- Tune hyperparameters:
  - Learning rate: `{0.01, 0.1, 0.5}`  
  - Iterations: `{500, 1000, 5000}`  
  - Hidden layer activation: `{sigmoid, relu}`  
- Implement evaluation metrics **from scratch**:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - ROC curve  

---

## 🧠 Theory

### XOR Problem
- A single-layer perceptron cannot solve XOR (not linearly separable).  
- Adding a **hidden layer** makes XOR solvable.  

### MLP Architecture
- **Input layer**: 2 features (x1, x2)  
- **Hidden layer**: 2 neurons with activation function f(z)  
- **Output layer**: 1 neuron with **sigmoid** activation  

### Forward Propagation
z(1) = XW(1) + b(1)  
h = f(z(1))  
z(2) = hW(2) + b(2)  
ŷ = σ(z(2))  

### Loss Function (MSE)
L = (1/2N) Σ (y - ŷ)^2  

### Backpropagation Update Rule
W ← W - η ∂L/∂W  

---

## ⚙️ Implementation Details
- Language: **Python (NumPy only)**  
- Training: Manual weight updates using gradient descent  
- Hyperparameter tuning: Grid search across learning rates, iterations, and activation functions  

---

## 📊 Evaluation Metrics

- **Accuracy**  
Accuracy = (TP + TN) / (TP + TN + FP + FN)

- **Precision**  
Precision = TP / (TP + FP)

- **Recall**  
Recall = TP / (TP + FN)

- **F1-score**  
F1 = 2 * Precision * Recall / (Precision + Recall)

- **ROC Curve**  
Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at different thresholds.



## 📈 Results

- With the best configuration (Sigmoid hidden layer, lr=0.1, iterations=5000), the model achieves:

| Metric      | Value |
|-------------|-------|
| Accuracy    | 1.0   |
| Precision   | 1.0   |
| Recall      | 1.0   |
| F1-score    | 1.0   |

- **Loss vs Iterations** → Decreases steadily  
- **ROC Curve** → Approaches the top-left corner (ideal classifier)  

---



## 📚 References
- Course materials (CSE412)  
- Goodfellow et al., *Deep Learning*  
- https://en.wikipedia.org/wiki/Backpropagation  



