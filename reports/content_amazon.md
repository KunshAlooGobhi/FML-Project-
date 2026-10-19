# Content model on Amazon products (from src/content_amazon.py)

Queries: 707 products in a leaf category shared with another product. Tuned on 353, reported on 354 held-out queries. Relevant = other products in the same leaf category (proxy; NOT user purchases - these files contain no users).

## Tuning grid (Table II)

|   max_features | ngram   |   vocab_size |   tune_NDCG@10 |
|---------------:|:--------|-------------:|---------------:|
|           5000 | (1, 1)  |         5000 |         0.5492 |
|           5000 | (1, 2)  |         5000 |         0.5594 |
|          20000 | (1, 1)  |        15774 |         0.5446 |
|          20000 | (1, 2)  |        20000 |         0.5552 |

## Held-out result

| model                                            |   NDCG@10 |   P@10 |   HR@10 |   MAP@10 |
|:-------------------------------------------------|----------:|-------:|--------:|---------:|
| random                                           |    0.0379 | 0.0367 |  0.2825 |   0.0133 |
| popularity (rating_count)                        |    0.0457 | 0.0393 |  0.2147 |   0.0211 |
| TF-IDF content (max_features=5000, ngram=(1, 2)) |    0.5668 | 0.4794 |  0.9209 |   0.4502 |

## Which text carries the signal

| text used                   |   NDCG@10 |   P@10 |   HR@10 |   MAP@10 |
|:----------------------------|----------:|-------:|--------:|---------:|
| title only                  |    0.5163 | 0.4322 |  0.8983 |   0.4005 |
| title + about               |    0.5248 | 0.4514 |  0.9011 |   0.4082 |
| title + about + description |    0.53   | 0.4523 |  0.9068 |   0.4097 |
| all text + review text      |    0.5668 | 0.4794 |  0.9209 |   0.4502 |

## Demo: similar products

**COOFANDY Men's Polo Shirts Short Sleeve Moisture Wicking Golf Shirt Fa** (Polos)
  - COOFANDY Knit Polo Shirts for Men Short Sleeve Textured Polo Shirt Sli (Polos)
  - BALENNZ Golf Polos for Men Quick-Dry Athletic Mens Polo Shirts Short S (Polos)
  - Men's Polo Shirts Performance Short Sleeve Golf Shirt Recycled Polyest (Polos)
  - 3 Pack Mens Polo Shirts Short Sleeve Sport Golf Tennis T Shirt Quick D (Polos)
  - Men's Long Sleeve Polo Shirts Casual Button Classic Work Shirt Cotton  (Polos)

**Rouen Mens Golf Shirt Moisture Wicking Dry Fit Performance Stripe Casu** (Polos)
  - ZITY 5 Pack Mens Polo Shirt Short Sleeve Sports Golf Tennis T-Shirt Mo (Polos)
  - M MAELREG Golf Shirts for Men Dry fit Sports Jacquard Lightweight Perf (Polos)
  - 3 Pack Mens Polo Shirts Short Sleeve Sport Golf Tennis T Shirt Quick D (Polos)
  - Men's Polo Shirts Performance Short Sleeve Golf Shirt Recycled Polyest (Polos)
  - ZENGVEE Mens Polo Shirts Moisture Wicking Short Sleeve Performance Qui (Polos)

**V VALANCH Mens Polo Shirts Short Sleeve Moisture Wicking Golf Polo Ath** (Polos)
  - ZITY 5 Pack Mens Polo Shirt Short Sleeve Sports Golf Tennis T-Shirt Mo (Polos)
  - ZITY 3 Pack Mens Polo Shirts Short Sleeve Golf Polo Moisture Wicking P (Polos)
  - 3 Pack Mens Polo Shirts Short Sleeve Sport Golf Tennis T Shirt Quick D (Polos)
  - ZITY 4 Pack Mens Polo Shirt Short Sleeve Sports Golf Tennis T-Shirt At (Polos)
  - ZITY Mens Polo Shirt Short Sleeve Sports Tactical T-Shirt Athletic Gol (Polos)
