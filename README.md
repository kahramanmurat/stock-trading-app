# Stock Trading App

This project aims to develop an end-to-end algorithmic trading strategy using machine learning techniques. The project involves selecting data sources, generating a unified dataset, performing data transformations, training ML models, defining and simulating trading strategies, and automating the solution. Docker deployments are also used to ensure reproducibility and scalability.

## Project Structure

```sh
stock-trading-app/
│
├── data/ # Placeholder for data storage
│
├── notebooks/ # Jupyter notebooks for various stages
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│ ├── strategy_simulation.ipynb
│ └── exploratory_analysis.ipynb
│
├── q_learning_trader/ # Q-learning trading strategy implementation
│
├── regression_based_strategy/ # Regression-based trading strategy implementation
│
├── trend_following_reinforcement_learning/ # Trend-following reinforcement learning strategy
│
├── trend_following_sma/ # Trend-following strategy using SMA
│
├── starter/ # Starter code and utilities
│
├── Dockerfile # Dockerfile for the project
│
├── requirements.txt # List of Python dependencies
├── main.py # Main script to run the application
├── LICENSE # License file
├── README.md # Project README file
└── .gitignore # Git ignore file```

## Getting Started

### Prerequisites

Ensure you have the following software installed:
- Python 3.8+
- Docker

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kahramanmurat/stock-trading-app.git
    cd stock-trading-app
    ```
2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
### Usage

1. Train the models and process the data:
    ```sh
    python main.py
    ```
It will run 4 type of strategy. It will print results and output the strategy plots in the folder.

- Q Learning Trader (Reinforcement Learning)
- Regression Based Strategy
    - Linear Regression
    - Logistic Regression
    - Random Forest
- Trend Following Strategy with Reinforcement Learning
- Trend Following Strategy with Simple Moving Average

### Running with Docker

1. Build the Docker image:
    ```sh
    docker build -t stock-trading-app .
    ```
2. Run the Docker Container:
    ```sh
    docker run -p 5000:5000 stock-trading-app
    ```
