import numpy as np
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge

JST_OFFSET_HOURS = 9


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cron_parts = df["cron"].str.split(expand=True)
    scheduled_hour_utc = cron_parts[1].astype(int) + cron_parts[0].astype(int) / 60
    df["scheduled_hour"] = (scheduled_hour_utc + JST_OFFSET_HOURS) % 24

    started_at_jst = pd.to_datetime(df["startedAt"], utc=True).dt.tz_convert("Asia/Tokyo")
    df["executed_hour"] = (
        started_at_jst.dt.hour
        + started_at_jst.dt.minute / 60
        + started_at_jst.dt.second / 3600
    )
    df = df.sort_values("startedAt").reset_index(drop=True)
    return df


def jst_hour_to_utc_cron(hour_value: float) -> tuple[int, int]:
    utc_hour_value = (hour_value - JST_OFFSET_HOURS) % 24
    cron_hour = int(utc_hour_value)
    cron_minute = int((utc_hour_value % 1) * 60)
    return cron_minute, cron_hour


def main():
    df = load_data("data/data.csv")

    X = df["executed_hour"].values.reshape(-1, 1)
    y = df["scheduled_hour"].values.reshape(-1, 1)

    reservoir = Reservoir(units=100, lr=0.4, sr=0.2, input_scaling=1 / 24)
    readout = Ridge(ridge=1e-9)
    model = reservoir >> readout
    model.fit(X, y, warmup=5)

    target_time = 12.0
    predicted_cron = model.run(np.array([[target_time]]))[0, 0]

    cron_minute, cron_hour = jst_hour_to_utc_cron(predicted_cron)

    print(f"{cron_minute} {cron_hour} * * *")


if __name__ == "__main__":
    main()
