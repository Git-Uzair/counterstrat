import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ZoneMapper:
    model: KNeighborsClassifier
    z_scale: float = 2.0

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    @classmethod
    def fit(cls, ticks: pl.DataFrame, z_scale: float = 2.0) -> "ZoneMapper":
        filtered = ticks.filter(
            pl.col("last_place_name").is_not_null() & (pl.col("last_place_name") != "")
        )
        if filtered.is_empty():
            raise ValueError("Cannot fit ZoneMapper: no valid tick rows")

        if filtered.height > 300_000:
            filtered = filtered.sample(n=300_000, seed=0)

        x_feat = filtered.select(
            [
                pl.col("X").cast(pl.Float64),
                pl.col("Y").cast(pl.Float64),
                (pl.col("Z").cast(pl.Float64) * z_scale),
            ]
        ).to_numpy()
        y_labels = filtered["last_place_name"].to_numpy()

        n_neighbors = min(5, filtered.height)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(x_feat, y_labels)
        return cls(model=knn, z_scale=z_scale)

    def zone(self, x: float, y: float, z: float) -> str:
        pred = self.model.predict([[float(x), float(y), float(z) * self.z_scale]])
        return str(pred[0])

    def zones(
        self,
        df: pl.DataFrame,
        x: str = "X",
        y: str = "Y",
        z: str = "Z",
    ) -> pl.Series:
        if df.is_empty():
            return pl.Series("zone", [], dtype=pl.String)

        x_scaled = df.select(
            [
                pl.col(x).cast(pl.Float64),
                pl.col(y).cast(pl.Float64),
                (pl.col(z).cast(pl.Float64) * self.z_scale),
            ]
        ).to_numpy()
        preds = self.model.predict(x_scaled)
        return pl.Series("zone", preds)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "ZoneMapper":
        p = Path(path)
        with p.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
