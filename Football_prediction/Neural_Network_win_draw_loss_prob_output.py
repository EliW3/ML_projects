import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from Getting_Data_and_Prepocessing import getData

df, elo_home_goal_spec, elo_away_goal_spec, elo_home_attack_spec, elo_away_attack_spec, elo_home_defense_spec, elo_away_defense_spec, elo_home, elo_away, elo_away_away_spec, elo_home_home_spec = getData("Austria", "Kosovo", timeframe=2020, current_time_year=2026, current_time_month=9, current_time_day=29)
teams = pd.concat([df["home_team"], df["away_team"]]).unique()
team_id = {team: i for i, team in enumerate(teams)}
df["team_id_home_team"] = df["home_team"].map(team_id)
df["team_id_away_team"] = df["away_team"].map(team_id)
home_team_id = {team: i for i, team in enumerate(df["home_team"].unique())}
df["home_team_id"] = df["home_team"].map(home_team_id)
away_team_id = {team: i for i, team in enumerate(df["away_team"].unique())}
df["away_team_id"] = df["away_team"].map(away_team_id)
tournament_id = {tournament: i for i, tournament in enumerate(df["tournament"].unique())}
df["tournament_id"] = df["tournament"].map(tournament_id)

class FootballPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.team_embedding = nn.Embedding(len(teams), 8)
        self.team_home_specific_embedding = nn.Embedding(len(teams), 8)
        self.team_away_specific_embedding = nn.Embedding(len(teams), 8)
        self.competition_embedding = nn.Embedding(len(tournament_id), 4)

        self.fc = nn.Sequential(
            nn.Linear(46, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, home_team_id, away_team_id, tournament_id, elo_feats):
        h = self.team_embedding(home_team_id)
        a = self.team_embedding(away_team_id)
        hs = self.team_home_specific_embedding(home_team_id)
        as1 = self.team_away_specific_embedding(away_team_id)
        c = self.competition_embedding(tournament_id)

        x = torch.cat([h, a, hs, as1, c, elo_feats], dim=1)
        return self.fc(x)
model = FootballPredictor()
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

elo_feats_fout = torch.tensor([[
            elo_home_goal_spec, elo_away_goal_spec,
            elo_home_attack_spec, elo_away_attack_spec,
            elo_home_defense_spec, elo_away_defense_spec,
            elo_home, elo_away, elo_home_home_spec, elo_away_away_spec
]], dtype=torch.float32) / 400.0
home = torch.tensor([team_id["Germany"]], dtype=torch.long)
away = torch.tensor([team_id["Brazil"]], dtype=torch.long)
tour = torch.tensor([tournament_id["FIFA World Cup"]], dtype=torch.long)

for epoch in range(10):
    total_loss = 0
    for i in range(len(df)):
        row = df.iloc[i]

        home = torch.tensor([row["team_id_home_team"]], dtype=torch.long)
        away = torch.tensor([row["team_id_away_team"]], dtype=torch.long)
        tour = torch.tensor([row["tournament_id"]], dtype=torch.long)

        ehgs = torch.tensor([row["ELO_goal_specific_home_before"]], dtype=torch.float32)
        eags = torch.tensor([row["ELO_goal_specific_away_before"]], dtype=torch.float32)
        ehas = torch.tensor([row["ELO_attack_home_before"]], dtype=torch.float32)
        eaas = torch.tensor([row["ELO_attack_away_before"]], dtype=torch.float32)
        ehds = torch.tensor([row["ELO_defense_home_before"]], dtype=torch.float32)
        eads = torch.tensor([row["ELO_defense_away_before"]], dtype=torch.float32)
        eh = torch.tensor([row["ELO_home_before"]], dtype=torch.float32)
        ea = torch.tensor([row["ELO_away_before"]], dtype=torch.float32)
        ehhs = torch.tensor([row["ELO_home_specific_before"]], dtype=torch.float32)
        eaas = torch.tensor([row["ELO_away_specific_before"]], dtype=torch.float32)

        elo_feats = torch.stack([
            ehgs, eags,
            ehas, eaas,
            ehds, eads,
            eh, ea, ehhs, eaas
        ], dim=1) / 400.0
        
        target = torch.tensor([row["Result"]], dtype=torch.long)

        logits = model(home, away, tour, elo_feats)
        loss = loss_func(logits, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} loss: {total_loss/len(df):.4f}")
home = torch.tensor([team_id["Germany"]], dtype=torch.long)
away = torch.tensor([team_id["Brazil"]], dtype=torch.long)
tour = torch.tensor([tournament_id["FIFA World Cup"]], dtype=torch.long)
model.eval()
with torch.no_grad():
    logits = model(home, away, tour, elo_feats_fout)
    probs = torch.softmax(logits, dim=1)
    print(probs, "([P(win home team), P(draw), P(win away team)")

def get_expected_values(self, home_win_betting_odds, draw_betting_odds, away_win_betting_odds, probs):
    home_win_expected_value = home_win_betting_odds * probs[0]
    draw_expected_value = draw_betting_odds * probs[1]
    away_win_expected_value = away_win_betting_odds[2]
    print(f"Home win Expected value: {home_win_expected_value}")
    print(f"Draw Expected value: {draw_expected_value}")
    print(f"Away win Expected value: {away_win_expected_value}")
