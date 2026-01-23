import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder 


# Load dataset
df = pd.read_csv("ipl.csv")
df.columns = df.columns.str.strip() 

# Clean dataset
df[['team1','team2','winner','venue']] = df[['team1','team2','winner','venue']].fillna('Unknown')

# Ensure numeric columns are numeric and no NaN
df['target_runs'] = pd.to_numeric(df['target_runs'], errors='coerce').fillna(0)
df['target_overs'] = pd.to_numeric(df['target_overs'], errors='coerce').fillna(20)          # default 20 overs


# Encode categorical features
le_team = LabelEncoder()
df['team1_enc'] = le_team.fit_transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])

le_venue = LabelEncoder()
df['venue_enc'] = le_venue.fit_transform(df['venue'])

le_winner = LabelEncoder()
df['winner_enc'] = le_winner.fit_transform(df['winner'])


# Simulate match features for ML & Using approximate values since match-level dataset
df['runs_left'] = df['target_runs'] - df['target_runs']*0.5
df['balls_left'] = df['target_overs']*6 - (df['target_overs']*6*0.5)
df['balls_left'] = df['balls_left'].replace(0,1)                            # for avoid zero balls
df['crr'] = (df['target_runs']*0.5) / (df['target_overs']*0.5)
df['rrr'] = df['runs_left'] / (df['balls_left']/6)
df['wickets_down'] = 5      # default value

# Drop any remaining rows with NaN in features
df = df.dropna(subset=['team1_enc','team2_enc','venue_enc','runs_left','balls_left','crr','rrr','wickets_down'])

# Features & target
X = df[['team1_enc','team2_enc','venue_enc','runs_left','balls_left','crr','rrr','wickets_down']]
y = df['winner_enc']


# Train model
model = LogisticRegression(max_iter=500)
model.fit(X, y)


# Streamlit UI
st.title("🏏 IPL Live Win Probability Predictor")

batting_team = st.selectbox("Batting Team:", le_team.classes_)
bowling_team = st.selectbox("Bowling Team:", le_team.classes_)
venue = st.selectbox("Venue:", le_venue.classes_)
target = st.number_input("Target Runs:", min_value=0, max_value=500, value=160)
current_score = st.number_input("Current Score:", min_value=0, max_value=500, value=50)
overs_completed = st.number_input("Overs Completed:", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
wickets_down = st.number_input("Wickets Down:", min_value=0, max_value=10, value=2)

if st.button("Predict Winning Probability"):

    # Safe feature calculations
    overs_completed = max(overs_completed, 0.1)
    runs_left = max(target - current_score, 0)
    balls_left = max((20*6) - (overs_completed*6), 1)
    crr = current_score / overs_completed
    rrr = runs_left / (balls_left/6)

    input_data = np.array([[ 
        le_team.transform([batting_team])[0],
        le_team.transform([bowling_team])[0],
        le_venue.transform([venue])[0],
        runs_left,
        balls_left,
        crr,
        rrr,
        wickets_down
    ]], dtype=float)

    # Convert any NaN to zero just in case
    input_data = np.nan_to_num(input_data)

    # Predict probability
    prob = model.predict_proba(input_data)[0][1]*100

    st.success(f"Winning Probability: {prob:.2f}%")
    if prob >= 50:
        st.info("Batting team is likely to WIN")
    else:
        st.warning("Batting team is likely to LOSE")

