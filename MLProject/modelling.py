import mlflow
import pandas as pd
import argparse
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

if os.getenv('GITHUB_ACTIONS'):
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
if not mlflow.active_run():
    mlflow.set_experiment("Eksperimen_rf_model_base")

mlflow.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
)


df = pd.read_csv('./StudentPerformance_preprocessing.csv')
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state)

print(f"Memulai training model dengan n_estimators: {args.n_estimators}...")
active_run = mlflow.active_run()
if active_run:
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print(f"RF R2 Score: {r2_score(y_test, rf_pred)}")
else:
    with mlflow.start_run(run_name="RandomForest_Base"):
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        print(f"RF R2 Score: {r2_score(y_test, rf_pred)}")
print("Selesai training model.")
