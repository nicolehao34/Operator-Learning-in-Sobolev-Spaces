
# Functional Analysis of Operator Learning: Quantitative Error Bounds in Sobolev Spaces
## Overview  
This project bridges **functional analysis** and **operator learning** to (more rigorously) analyze how deep neural networks approximate nonlinear operators between Sobolev spaces. 

The goal of this project is to characterize/study the interplay between neural network architecture, optimization dynamics, and approximation error in high-dimensional function spaces, with applications to learning solution operators of PDEs.  

Work in Progress! I will add the code for numerical experiments soon :)
## Update History

This section chronicles the major updates and milestones of the project:

- **December 11, 2025**: Minor update, compiled all scripts to train_fno.py and parameter_sweep.py. Kept solver.py for future references.
- **May 27, 2025**: Initial repository structure established with numerics and theory directories
- **June 6, 2025**: Released first iteration of numerical implementations, updated project documentation, and reorganized repository structure for improved maintainability

---

## Some background info on Operator Learning
What's operator learning exactly?

So, in normal machine learning, you often try to learn a function, like predicting the price of a house based on its size. 

**One input → one output.**

But in science (like physics or climate modeling), we often care about something more complex: what happens to an entire shape, signal, or function when it goes through a system. 

For example:

Given the temperature across the earth today (a function), predict the temperature tomorrow (another function).

Given how a material is stretched, predict how it bends or cracks.

This is where operator learning comes in! An operator is like a machine that takes in a whole function and gives you another function. So:

Instead of **input: a number → output: a number**

It's **input: a function → output: a function**

Operator learning means using a neural network to learn a machine (the operator) from examples, so you can use it to make predictions in the future.


--- 

## Sections
### 1. **Theoretical Contributions**  
- **Operator Approximation**: Prove universal approximation theorems for neural networks mapping between Sobolev spaces \( H^s(D) \to H^t(D') \).  
- **Error Bounds**: Establish scaling laws for network depth/width and their impact on generalization in Sobolev norms.  

### 2. **Numerical Experiments**  
- **Burgers Equation**: Learn the operator mapping initial conditions to PDE solutions.  
- **Width/Depth Scaling**: Validate theoretical depth-width-error tradeoffs.  


---

##  Setup  & Dependencies  
- **Python**: For numerical experiments (PyTorch, NumPy, Scipy).  
- Install dependecies first using `pip install -r requirements.txt`

---
## Repo Structure 
-  `theory/`
    - `results.pdf`: proofs and theoretical background.  

- `numerics/`: use this set of scripts to reproduce experiments and numerical results.  

    - `solver.py`: Spectral solver for Burgers equation (kept for reference)

    - `train_fno.py`: Compiled training script with full training loop

    - `parameter_sweep.py`: Script for hyperparameter experiments and depth/width scaling studies

