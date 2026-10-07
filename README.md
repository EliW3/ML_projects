# ML_projects

# 1. Football prediction Neural Network
Step 1: Getting data and engineering useful features to improve accuracy
        Features:
                  is neutral (true/false),
                  general ELO,
                  ELO goal specific,
                  ELO attack/defense specific,
                  ELO home/away specific,
                  win streak,
                  no-loss streak,
                  rolling goals scored,
                  rolling goals conceded,
                  rolling goals difference,
                  form points,
                  clean sheets,
                  team embedding,
                  team home/away specific embedding,
                  tournament embedding
Step 2: Setting up Neural Network and training (Standard 30 epochs)
        Size: 60 * 128 * 64 * 32 * 3
        Loss: Cross entropy loss (loss function for softmax/multi output probabilities
Step 3: Predicting matches
        Output form: Softmax (Win probability, draw probability, loss probability)
Step 4: Backtesting on 3143 matches
        Accuracy: ~56%
        Log loss: ~0.94
        Brier: ~0.55
                  
                  
