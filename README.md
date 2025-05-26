Source code of *Spatiotemporally adaptive optimization of** **nonpharmaceutical interventions for influenza via a SEIQR-Dueling DQN framework*

# Project Description

This project implements a SEIQR-Dueling DQN framework for spatiotemporally adaptive optimization of nonpharmaceutical interventions (NPIs) for influenza. It aims to address the challenges of balancing infection control and socioeconomic costs while accounting for regional heterogeneity in influenza transmission dynamics.

# Usage
### Data Preparation
1. Ensure that the climatic data (`weatherSichuan.xlsx`) is available in the root directory.
2. Modify the data preprocessing scripts (`SEIQR.py` and `SEIQR_random.py`) if necessary to match your data structure.

### SEIQR Model
1. **Run the SEIQR Model**:
    ```bash
    python SEIQR.py
    ```
    This script initializes the SEIQR model and runs simulations based on the provided climatic and demographic data.

2. **Sensitivity Analysis**:
    ```bash
    python sensitivity.py
    ```
    This script performs sensitivity analysis on the SEIQR model parameters to assess their impact on the model outcomes.

### Reinforcement Learning (DQN)
1. **Train the DQN Model**:
    ```bash
    python RL/trainDQN.py
    ```
    This script trains the Dueling DQN agent using the SEIQR model's output as the environment. It saves the trained model weights and logs the training process.

2. **Evaluate the DQN Model**:
    ```bash
    python RL/testDQN.py
    ```
    This script evaluates the trained DQN model's performance by comparing it with the threshold-based NPI strategy. It generates performance metrics and visualizations.

3. **Additional Testing**:
    ```bash
    python RL/test2DQN.py
    ```
    This script provides additional testing scenarios for the DQN model, including different initial conditions and parameter variations.

### Randomized Reinforcement Learning (DQN)
1. **Train the Randomized DQN Model**:
    ```bash
    python RL_random/trainDQN.py
    ```
    This script trains the randomized Dueling DQN agent, which incorporates random perturbations in the action space to assess robustness.

2. **Evaluate the Randomized DQN Model**:
    ```bash
    python RL_random/testDQN.py
    ```
    This script evaluates the performance of the randomized DQN model and compares it with the standard DQN model.

3. **Plot Results**:
    ```bash
    python RL_random/testDQN_plot.py
    ```
    This script generates plots to visualize the performance of the randomized DQN model.
