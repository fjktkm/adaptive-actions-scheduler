import numpy as np
import pandas as pd
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cron_parts = df["cron"].str.split(expand=True)
    df["scheduled_hour"] = cron_parts[1].astype(int) + cron_parts[0].astype(int) / 60
    df["executed_hour"] = (
        pd.to_datetime(df["startedAt"]).dt.hour
        + pd.to_datetime(df["startedAt"]).dt.minute / 60
        + pd.to_datetime(df["startedAt"]).dt.second / 3600
    )
    return df


def build_model():
    rpy.set_seed(42)
    reservoir = Reservoir(units=100, lr=0.5, sr=0.9, input_scaling=1 / 24)
    readout = Ridge(ridge=1e-7)
    return reservoir >> readout


def main():
    df = load_data("data/data.csv")

    X = df["executed_hour"].values.reshape(-1, 1)
    y = df["scheduled_hour"].values.reshape(-1, 1)

    model = build_model()
    model.fit(X, y, warmup=1)

    target_time = 3.0
    predicted_cron = model.run(np.array([[target_time]]))[0, 0]

    cron_hour = int(predicted_cron)
    cron_minute = int((predicted_cron % 1) * 60)

    print(f"{cron_minute} {cron_hour} * * *")


if __name__ == "__main__":
    main()
