# CSE337 Final Project: Cricket Shot Selection Optimizer

### This project utilizes a DQN algorithm to train a cricket batsman to maximize the number of runs in an over without losing their wicket. We created our own custom cricket environment from a publically available cricket dataset located on [Kaggle.](https://www.kaggle.com/datasets/piyushsharma18/cricket-shot-selection?resource=download)



### To run this project:
1. Install the project's dependencies from the requirements.txt file: `pip install -r requirements.txt`
2. Build the cricket environment: `python prediction_model_new.py`
3. Train the agent with DQN: `python dqn_with_graphs.py`
4. (Optional) Train a baseline defensive agent for comparison `python dqn_defense_model.py`
5. Visualize the agent's decsions with a MuJoCo simulation: `python simulate.py`

### Additionally, several graphs will be generated from step 3, which will show various metrics throughout the training. 