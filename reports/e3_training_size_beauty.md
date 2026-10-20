# E3: training-size sweep (real Amazon Beauty data, validation only; from src/e3_sweep.py)

Evaluated on the same 5,446 users in every row.

NDCG@10 by share of training data used:

| model         |     25% |     50% |    100% |
|:--------------|--------:|--------:|--------:|
| ALS           | 0.00336 | 0.00763 | 0.0171  |
| Content-based | 0.0143  | 0.01599 | 0.0187  |
| Item-kNN      | 0.00416 | 0.00745 | 0.01785 |
| Popularity    | 0.00579 | 0.00564 | 0.00569 |

Detail:

| training data   |   interactions | model         |   NDCG@10 |   HR@10 |
|:----------------|---------------:|:--------------|----------:|--------:|
| 25%             |          34680 | Popularity    |   0.00579 | 0.02552 |
| 25%             |          34680 | Item-kNN      |   0.00416 | 0.01653 |
| 25%             |          34680 | ALS           |   0.00336 | 0.01194 |
| 25%             |          34680 | Content-based |   0.0143  | 0.04903 |
| 50%             |          69361 | Popularity    |   0.00564 | 0.02516 |
| 50%             |          69361 | Item-kNN      |   0.00745 | 0.02369 |
| 50%             |          69361 | ALS           |   0.00763 | 0.02516 |
| 50%             |          69361 | Content-based |   0.01599 | 0.05454 |
| 100%            |         138723 | Popularity    |   0.00569 | 0.02497 |
| 100%            |         138723 | Item-kNN      |   0.01785 | 0.0538  |
| 100%            |         138723 | ALS           |   0.0171  | 0.05215 |
| 100%            |         138723 | Content-based |   0.0187  | 0.06427 |
