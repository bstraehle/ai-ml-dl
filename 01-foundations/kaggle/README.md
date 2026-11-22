---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Kaggle

Kaggle competitions are a good way to practice the whole machine learning workflow on real data, and to compare your results with others. This folder has the data for two classic beginner competitions.

**Roles:** data scientist, ML engineer.
**Prerequisites:** [Machine learning](../machine-learning/)

## Key concepts

- **Start with a baseline.** A simple model (for example, everyone survives, or the median price) sets the bar every later idea must beat.
- **Explore before modeling.** Look at distributions, missing values and relationships with the target; most gains come from understanding the data.
- **Feature engineering** turns raw columns into useful signals: extracting titles from names, family size, combining areas, encoding categories.
- **Validate locally.** Use cross-validation on the training set instead of submitting repeatedly. The public leaderboard uses a small sample, and tuning to it overfits.
- **Gradient-boosted trees** (XGBoost, LightGBM, CatBoost) are strong defaults for tabular competitions.
- **Learn from others.** Read top public notebooks and discussions after each competition.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Titanic: binary classification with tabular data](https://www.kaggle.com/competitions/titanic/) | Competition | Beginner | Free |
| [House prices: regression with tabular data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) | Competition | Beginner | Free |
| [Kaggle Learn](https://www.kaggle.com/learn) | Courses | Beginner | Free |
| [My Kaggle profile](https://www.kaggle.com/bstraehle) | Profile | | |

## Hands-on

| Folder | Contents |
|---|---|
| [titanic/](titanic/) | Training and test data, and an example submission |
| [house-prices/](house-prices/) | Training and test data, a data description, and an example submission |

## Next steps

- [Deep learning](../deep-learning/)
- [MLOps](../../09-ops-and-deployment/machine-learning-eng/): take a model beyond a notebook
