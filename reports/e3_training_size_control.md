# E3: training-size sweep (control data, validation only; from src/e3_sweep.py)

Evaluated on the same 4,865 users in every row.

NDCG@10 by share of training data used:

| model      |     25% |     50% |    100% |
|:-----------|--------:|--------:|--------:|
| ALS        | 0.07002 | 0.10982 | 0.1692  |
| Item-kNN   | 0.02817 | 0.11743 | 0.18278 |
| Popularity | 0.05357 | 0.05574 | 0.05925 |

Detail:

| training data   |   interactions | model      |   NDCG@10 |   HR@10 |
|:----------------|---------------:|:-----------|----------:|--------:|
| 25%             |          21862 | Popularity |   0.05357 | 0.23309 |
| 25%             |          21862 | Item-kNN   |   0.02817 | 0.13176 |
| 25%             |          21862 | ALS        |   0.07002 | 0.29435 |
| 50%             |          43725 | Popularity |   0.05574 | 0.23844 |
| 50%             |          43725 | Item-kNN   |   0.11743 | 0.41377 |
| 50%             |          43725 | ALS        |   0.10982 | 0.41336 |
| 100%            |          87451 | Popularity |   0.05925 | 0.25242 |
| 100%            |          87451 | Item-kNN   |   0.18278 | 0.55581 |
| 100%            |          87451 | ALS        |   0.1692  | 0.5369  |
