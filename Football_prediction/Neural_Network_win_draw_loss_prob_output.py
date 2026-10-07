import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

TRAIN_START = 2015
TEST_START = 2023
TEST_END = 2026

K = 24
K_GOALS = 10

df = pd.read_csv(URL)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

current_elo = defaultdict(lambda: 1500.0)
home_elo = defaultdict(lambda: 1500.0)
away_elo = defaultdict(lambda: 1500.0)
goal_home_elo = defaultdict(lambda: 1500.0)
goal_away_elo = defaultdict(lambda: 1500.0)
attack_elo = defaultdict(lambda: 1500.0)
defense_elo = defaultdict(lambda: 1500.0)

win_streak = defaultdict(int)
not_losing_streak = defaultdict(int)
goals_scored_history = defaultdict(list)
goals_conceded_history = defaultdict(list)
form_history = defaultdict(list)
clean_sheet_history = defaultdict(list)

features = []

for row in df.itertuples():
    h, a = row.home_team, row.away_team
    hs, aws = row.home_score, row.away_score
    date = row.date

    rh, ra = current_elo[h], current_elo[a]
    rhs, ras = home_elo[h], away_elo[a]
    rgh, rga = goal_home_elo[h], goal_away_elo[a]
    rah, raa = attack_elo[h], attack_elo[a]
    rdh, rda = defense_elo[h], defense_elo[a]

    e_home = 1 / (1 + 10 ** ((ra - rh) / 400))
    e_home_spec = 1 / (1 + 10 ** ((ras - rhs) / 400))

    exp_home_goals = 1.4 * 10 ** ((rah - rda) / 400)
    exp_away_goals = 1.4 * 10 ** ((raa - rdh) / 400)

    gs_h = goals_scored_history[h][-5:]
    gc_h = goals_conceded_history[h][-5:]
    gs_a = goals_scored_history[a][-5:]
    gc_a = goals_conceded_history[a][-5:]

    pts_h = form_history[h][-5:]
    pts_a = form_history[a][-5:]
    cs_h = clean_sheet_history[h][-5:]
    cs_a = clean_sheet_history[a][-5:]

    features.append({
        "date": date,
        "home_team": h,
        "away_team": a,
        "tournament": row.tournament,
        "home_score": hs,
        "away_score": aws,
        "Result": 0 if hs > aws else 1 if hs == aws else 2,

        "ELO_home": rh,
        "ELO_away": ra,
        "ELO_home_specific": rhs,
        "ELO_away_specific": ras,
        "ELO_goal_home": rgh,
        "ELO_goal_away": rga,
        "ELO_attack_home": rah,
        "ELO_attack_away": raa,
        "ELO_defense_home": rdh,
        "ELO_defense_away": rda,

        "Winning_streak_home": win_streak[h],
        "Winning_streak_away": win_streak[a],
        "Not_losing_streak_home": not_losing_streak[h],
        "Not_losing_streak_away": not_losing_streak[a],

        "Rolling_goals_scored_home": np.mean(gs_h) if gs_h else np.nan,
        "Rolling_goals_conceded_home": np.mean(gc_h) if gc_h else np.nan,
        "Rolling_goals_scored_away": np.mean(gs_a) if gs_a else np.nan,
        "Rolling_goals_conceded_away": np.mean(gc_a) if gc_a else np.nan,

        "Rolling_goal_difference_home": np.sum(gs_h) - np.sum(gc_h) if gs_h else np.nan,
        "Rolling_goal_difference_away": np.sum(gs_a) - np.sum(gc_a) if gs_a else np.nan,

        "Form_points_home": np.sum(pts_h) if pts_h else np.nan,
        "Form_points_away": np.sum(pts_a) if pts_a else np.nan,

        "Clean_sheets_home": np.sum(cs_h) if cs_h else np.nan,
        "Clean_sheets_away": np.sum(cs_a) if cs_a else np.nan,
    })

    gd = hs - aws
    mult = 1.0 if gd == 0 else np.log(abs(gd) + 1) * 1.75 / (1.75 + 0.00175 * abs(rh - ra))

    if gd > 0:
        sh, sa = 1.0, 0.0
    elif gd < 0:
        sh, sa = 0.0, 1.0
    else:
        sh, sa = 0.5, 0.5

    new_rh = rh + K * (sh - e_home)
    new_ra = ra + K * (sa - (1 - e_home))

    new_rhs = rhs + K * (sh - e_home_spec)
    new_ras = ras + K * (sa - (1 - e_home_spec))

    new_rgh = rgh + K * mult * (sh - e_home)
    new_rga = rga + K * mult * (sa - (1 - e_home))

    diff_h = hs - exp_home_goals
    diff_a = aws - exp_away_goals

    new_rah = rah + K_GOALS * diff_h
    new_rda = rda - K_GOALS * diff_h
    new_raa = raa + K_GOALS * diff_a
    new_rdh = rdh - K_GOALS * diff_a

    current_elo[h], current_elo[a] = new_rh, new_ra
    goal_home_elo[h], goal_away_elo[a] = new_rgh, new_rga
    attack_elo[h], attack_elo[a] = new_rah, new_raa
    defense_elo[h], defense_elo[a] = new_rdh, new_rda

    if not getattr(row, "neutral", False):
        home_elo[h] = new_rhs
        away_elo[a] = new_ras

    goals_scored_history[h].append(hs)
    goals_conceded_history[h].append(aws)
    goals_scored_history[a].append(aws)
    goals_conceded_history[a].append(hs)

    clean_sheet_history[h].append(int(aws == 0))
    clean_sheet_history[a].append(int(hs == 0))

    if gd > 0:
        form_history[h].append(3)
        form_history[a].append(0)
        win_streak[h] += 1
        win_streak[a] = 0
        not_losing_streak[h] += 1
        not_losing_streak[a] = 0
    elif gd == 0:
        form_history[h].append(1)
        form_history[a].append(1)
        win_streak[h] = 0
        win_streak[a] = 0
        not_losing_streak[h] += 1
        not_losing_streak[a] += 1
    else:
        form_history[h].append(0)
        form_history[a].append(3)
        win_streak[h] = 0
        win_streak[a] += 1
        not_losing_streak[h] = 0
        not_losing_streak[a] += 1

df = pd.DataFrame(features)

df = df[
    (df["date"] >= f"{TRAIN_START}-01-01") &
    (df["date"] < f"{TEST_END}-12-31")
].reset_index(drop=True)

df = df.dropna().reset_index(drop=True)

train_df = df[df["date"] < f"{TEST_START}-01-01"].copy()

test_df = df[
    (df["date"] >= f"{TEST_START}-01-01") &
    (df["date"] < f"{TEST_END}-01-01")
].copy()


train_teams = pd.concat([
    train_df["home_team"],
    train_df["away_team"]
]).unique()

train_tournaments = train_df["tournament"].unique()

team_id = {
    team: i
    for i, team in enumerate(train_teams)
}

tournament_id = {
    tournament: i
    for i, tournament in enumerate(train_tournaments)
}

df["home_id"] = df["home_team"].map(team_id)
df["away_id"] = df["away_team"].map(team_id)
df["tournament_id"] = df["tournament"].map(tournament_id)
df = df.dropna().reset_index(drop=True)

elo_columns = [
    "ELO_goal_home",
    "ELO_goal_away",
    "ELO_attack_home",
    "ELO_attack_away",
    "ELO_defense_home",
    "ELO_defense_away",
    "ELO_home",
    "ELO_away",
    "ELO_home_specific",
    "ELO_away_specific"
]
running_columns = [
    "Winning_streak_home",
    "Winning_streak_away",
    "Not_losing_streak_home",
    "Not_losing_streak_away",
    "Rolling_goals_scored_home",
    "Rolling_goals_conceded_home",
    "Rolling_goals_scored_away",
    "Rolling_goals_conceded_away",
    "Rolling_goal_difference_home",
    "Rolling_goal_difference_away",
    "Form_points_home",
    "Form_points_away",
    "Clean_sheets_home",
    "Clean_sheets_away"
]

class FootballPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.85)

        self.team_embedding = nn.Embedding(len(train_teams), 8)
        self.team_home_specific_embedding = nn.Embedding(len(train_teams), 8)
        self.team_away_specific_embedding = nn.Embedding(len(train_teams), 8)
        self.competition_embedding = nn.Embedding(len(train_tournaments), 4)

        self.fc = nn.Sequential(
            nn.Linear(60, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 3)
        )

    def forward(self, home, away, tournament, elo, running):
        h = self.team_embedding(home)
        a = self.team_embedding(away)
        hs = self.team_home_specific_embedding(home)
        aw = self.team_away_specific_embedding(away)
        c = self.competition_embedding(tournament)

        x = torch.cat([h, a, hs, aw, c, elo, running], dim=1)
        return self.fc(x) / self.temperature

train_df = df[df["date"] < f"{TEST_START}-01-01"].copy()
test_df = df[
    (df["date"] >= f"{TEST_START}-01-01") &
    (df["date"] < f"{TEST_END}-01-01")
].copy()

X_home = torch.tensor(train_df["home_id"].values, dtype=torch.long)
X_away = torch.tensor(train_df["away_id"].values, dtype=torch.long)
X_tournament = torch.tensor(train_df["tournament_id"].values, dtype=torch.long)

X_elo = torch.tensor(
    train_df[elo_columns].values,
    dtype=torch.float32
) / 400.0

X_running = torch.tensor(
    train_df[running_columns].values,
    dtype=torch.float32
)

y = torch.tensor(
    train_df["Result"].values,
    dtype=torch.long
)

dataset = TensorDataset(
    X_home,
    X_away,
    X_tournament,
    X_elo,
    X_running,
    y
)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

model = FootballPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    total_loss = 0

    for home, away, tournament, elo, running, target in loader:
        logits = model(
            home,
            away,
            tournament,
            elo,
            running
        )

        loss = loss_func(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1:02d} "
        f"Loss: {total_loss / len(loader):.4f}"
    )

def get_latest_team_state(team):
    home_matches = df[df["home_team"] == team]
    away_matches = df[df["away_team"] == team]

    candidates = []

    if len(home_matches) > 0:
        candidates.append(("home", home_matches.iloc[-1]))

    if len(away_matches) > 0:
        candidates.append(("away", away_matches.iloc[-1]))

    if not candidates:
        raise ValueError(f"No historical matches found for {team}")

    role, row = max(
        candidates,
        key=lambda x: x[1]["date"]
    )

    if role == "home":
        suffix = "home"
    else:
        suffix = "away"

    return {
        "ELO_goal": row[f"ELO_goal_{suffix}"],
        "ELO_attack": row[f"ELO_attack_{suffix}"],
        "ELO_defense": row[f"ELO_defense_{suffix}"],
        "ELO": row[f"ELO_{suffix}"],
        "ELO_specific": row[f"ELO_{suffix}_specific"],

        "Winning_streak": row[f"Winning_streak_{suffix}"],
        "Not_losing_streak": row[f"Not_losing_streak_{suffix}"],

        "Rolling_goals_scored": row[
            f"Rolling_goals_scored_{suffix}"
        ],

        "Rolling_goals_conceded": row[
            f"Rolling_goals_conceded_{suffix}"
        ],

        "Rolling_goal_difference": row[
            f"Rolling_goal_difference_{suffix}"
        ],

        "Form_points": row[
            f"Form_points_{suffix}"
        ],

        "Clean_sheets": row[
            f"Clean_sheets_{suffix}"
        ],
    }

def predict_matches(matches):
    model.eval()
    predictions = []

    with torch.no_grad():
        for home_team, away_team, competition in matches:
            if home_team not in team_id:
                raise ValueError(f"Unknown home team: {home_team}")
            if away_team not in team_id:
                raise ValueError(f"Unknown away team: {away_team}")
            if competition not in tournament_id:
                raise ValueError(f"Unknown competition: {competition}")

            h = get_latest_team_state(home_team)
            a = get_latest_team_state(away_team)

            elo = torch.tensor([[
                h["ELO_goal"],
                a["ELO_goal"],
                h["ELO_attack"],
                a["ELO_attack"],
                h["ELO_defense"],
                a["ELO_defense"],
                h["ELO"],
                a["ELO"],
                h["ELO_specific"],
                a["ELO_specific"]
            ]], dtype=torch.float32) / 400.0

            running = torch.tensor([[
                h["Winning_streak"],
                a["Winning_streak"],
            
                h["Not_losing_streak"],
                a["Not_losing_streak"],
            
                h["Rolling_goals_scored"],
                h["Rolling_goals_conceded"],
            
                a["Rolling_goals_scored"],
                a["Rolling_goals_conceded"],
            
                h["Rolling_goal_difference"],
                a["Rolling_goal_difference"],
            
                h["Form_points"],
                a["Form_points"],
            
                h["Clean_sheets"],
                a["Clean_sheets"]
            ]], dtype=torch.float32)

            home = torch.tensor([team_id[home_team]])
            away = torch.tensor([team_id[away_team]])
            tournament = torch.tensor([tournament_id[competition]])

            probs = torch.softmax(
                model(home, away, tournament, elo, running),
                dim=1
            )[0]

            predictions.append({
                "home_team": home_team,
                "away_team": away_team,
                "competition": competition,
                "home_win_probability": probs[0].item(),
                "draw_probability": probs[1].item(),
                "away_win_probability": probs[2].item(),
                "prediction": ["Home", "Draw", "Away"][probs.argmax().item()]
            })

    return pd.DataFrame(predictions)

def backtest(model, test_df):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            elo = torch.tensor([[
                row["ELO_goal_home"],
                row["ELO_goal_away"],
                row["ELO_attack_home"],
                row["ELO_attack_away"],
                row["ELO_defense_home"],
                row["ELO_defense_away"],
                row["ELO_home"],
                row["ELO_away"],
                row["ELO_home_specific"],
                row["ELO_away_specific"]
            ]], dtype=torch.float32) / 400.0

            running = torch.tensor([[
    row["Winning_streak_home"],
    row["Winning_streak_away"],

    row["Not_losing_streak_home"],
    row["Not_losing_streak_away"],

    row["Rolling_goals_scored_home"],
    row["Rolling_goals_conceded_home"],

    row["Rolling_goals_scored_away"],
    row["Rolling_goals_conceded_away"],

    row["Rolling_goal_difference_home"],
    row["Rolling_goal_difference_away"],

    row["Form_points_home"],
    row["Form_points_away"],

    row["Clean_sheets_home"],
    row["Clean_sheets_away"]
]], dtype=torch.float32)

            home = torch.tensor([row["home_id"]], dtype=torch.long)
            away = torch.tensor([row["away_id"]], dtype=torch.long)
            tournament = torch.tensor(
                [row["tournament_id"]],
                dtype=torch.long
            )

            probs = torch.softmax(
                model(home, away, tournament, elo, running),
                dim=1
            )[0]

            predictions.append(probs.numpy())
            actual.append(int(row["Result"]))

    predictions = np.array(predictions)
    actual = np.array(actual)

    return predictions, actual

def evaluate_predictions(predictions, actual):
    predictions = np.clip(
        predictions,
        1e-15,
        1 - 1e-15
    )

    predictions = predictions / predictions.sum(axis=1, keepdims=True)

    logloss = -np.mean(
        np.log(
            predictions[
                np.arange(len(actual)),
                actual
            ]
        )
    )

    predicted_class = np.argmax(
        predictions,
        axis=1
    )

    accuracy = np.mean(
        predicted_class == actual
    )

    one_hot = np.eye(3)[actual]

    brier = np.mean(
        np.sum(
            (predictions - one_hot) ** 2,
            axis=1
        )
    )

    confidence = predictions.max(axis=1)
    correct = (predicted_class == actual).astype(float)

    bins = np.linspace(0, 1, 11)
    calibration = []

    for i in range(10):
        if i == 9:
            mask = (
                (confidence >= bins[i]) &
                (confidence <= bins[i + 1])
            )
        else:
            mask = (
                (confidence >= bins[i]) &
                (confidence < bins[i + 1])
            )

        if mask.sum() > 0:
            calibration.append({
                "bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}",
                "predicted": confidence[mask].mean(),
                "actual": correct[mask].mean(),
                "count": int(mask.sum())
            })

    calibration = pd.DataFrame(calibration)

    print(f"\nMatches:  {len(actual)}")
    print(f"Log loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Brier:    {brier:.4f}")
    print("\nCalibration:")
    print(calibration.to_string(index=False))

    return {
        "logloss": logloss,
        "accuracy": accuracy,
        "brier": brier,
        "calibration": calibration
    }

predictions, actual = backtest(model, test_df)

results = evaluate_predictions(
    predictions,
    actual
)

backtest_df = test_df[
    ["date", "home_team", "away_team", "tournament",
     "home_score", "away_score", "Result"]
].copy()

backtest_df["home_win_probability"] = predictions[:, 0]
backtest_df["draw_probability"] = predictions[:, 1]
backtest_df["away_win_probability"] = predictions[:, 2]
backtest_df["prediction"] = np.argmax(predictions, axis=1)

backtest_df["prediction"] = backtest_df["prediction"].map({
    0: "Home",
    1: "Draw",
    2: "Away"
})

print("\nBacktest:")
print(backtest_df.head(20).to_string(index=False))

matches = [
    ("Northern Ireland", "Austria", "UEFA Nations League"),
    ("Germany", "Netherlands", "UEFA Nations League"),
    ("France", "Italy", "UEFA Nations League")
]

future_predictions = predict_matches(matches)

print("\nPredictions:")
print(future_predictions.to_string(index=False))
