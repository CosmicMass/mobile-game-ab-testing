# data/

The dataset is a public download and is **not committed to this repository**.
Everything else in this directory is gitignored.

## Getting it

1. Download **Mobile Games: A/B Testing** from Kaggle:
   <https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing>
   (published by Aurelia Sui / `yufengsui`, July 2019).
2. Put `cookie_cats.csv` in this directory.

```
data/cookie_cats.csv
```

`examples/run_analysis.py` picks it up from there automatically. Without it,
the script falls back to generated stand-in data and says so, loudly, at the
top and bottom of its output.

## What the file contains

90,189 rows, one per player, from an A/B test on the mobile puzzle game
Cookie Cats. The game shows a "gate" that forces a wait (or an in-app
purchase) before the player can continue; the experiment moved that gate from
level 30 to level 40.

| column | type | meaning |
|---|---|---|
| `userid` | int | Unique player id. |
| `version` | str | Experiment arm: `gate_30` (control) or `gate_40` (treatment). |
| `sum_gamerounds` | int | Rounds played in the first 14 days after install. Extremely right-skewed -- one player logged 49,854. |
| `retention_1` | bool | Did the player come back 1 day after installing? |
| `retention_7` | bool | Did the player come back 7 days after installing? |

`loader.load_cookie_cats` validates all of this on the way in and refuses a
file that does not match.
