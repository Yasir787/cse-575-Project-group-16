# Analysis of Late Generalization in Neural Architectures
**An Empirical Study of Grokking and Optimization Dynamics in Deep Neural Networks**

---

## 📌 Overview
This project investigates the phenomenon of **late generalization (commonly referred to as *grokking*)** in deep neural networks, where models achieve near-perfect training performance long before exhibiting strong generalization on unseen data.

Through controlled experiments, this study analyzes **training dynamics, optimization behavior, and generalization delay** across different neural architectures and training regimes.

The goal is to better understand *why* and *when* generalization emerges beyond memorization.

---

## 🔬 Research Questions
- Under what conditions does late generalization occur in neural networks?
- How do **training duration, model capacity, and regularization** influence delayed generalization?
- How do optimization dynamics differ between early-generalizing and late-generalizing models?
- Do implicit neural representations exhibit different grokking behavior compared to standard architectures?

---

## 🧪 Experimental Setup
- **Model Types:** Standard feedforward neural networks and implicit neural representations  
- **Training Regime:** Extended training beyond initial convergence  
- **Evaluation:** Monitoring training loss, validation loss, and generalization gap over time  
- **Metrics:** Accuracy, loss trajectories, convergence behavior  

Experiments are designed to isolate **generalization emergence** from simple memorization.

---

## 📊 Key Observations
- Models often achieve **near-zero training loss** long before generalization improves.
- Extended training reveals a **sudden transition phase** where validation performance rapidly increases.
- Generalization delay is strongly influenced by **optimization dynamics and regularization**, not just model size.
- Implicit representations show distinct convergence patterns compared to standard architectures.

> These findings align with recent theoretical discussions on grokking and neural optimization.

---

## 🧠 Why This Matters
Understanding late generalization has implications for:
- Training efficiency in deep learning
- Model selection and stopping criteria
- Interpreting overparameterized neural networks
- Bridging theory and practice in modern ML systems

This work contributes to the broader discussion on **how deep models learn beyond memorization**.

---

## 🛠 Tech Stack
- **Language:** Python  
- **Deep Learning:** PyTorch / TensorFlow  
- **Data Analysis:** NumPy  
- **Visualization:** Matplotlib  

---

## 📁 Repository Structure
