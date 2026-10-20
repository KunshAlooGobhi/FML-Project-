# 5-minute presentation script (8 slides, about 35 seconds each)

Speakers: Rishit and Kunsh. Say "NDCG" as "a score from 0 to 1 that rewards correct products near the top".

**Slide 1: Title and question** (Rishit)
"Our project is a recommender for online shopping. The question: can a model rank the products a shopper will buy next better than the simple best-seller list? We tested this on real Amazon Beauty reviews: shampoo, mascara, perfume and similar products. 22,000 shoppers, 12,000 products, 2002 to 2014."

**Slide 2: Why it is hard** (Rishit)
"Three problems: most shoppers buy very few items, so the data is mostly empty; random train/test splits leak the future and give fake-good scores; and best-seller baselines are strong. So we split by time, froze every setting before testing, and scored the final test period exactly once."

**Slide 3: The models** (Kunsh)
"We compared: best-sellers, item-based collaborative filtering ('people who bought this also bought'), SVD, ALS and BPR (learn hidden tastes from purchases), a text model that matches product descriptions, and a hybrid that blends collaborative and text scores. For brand-new products with no purchases, the hybrid uses the text only."

**Slide 4: Main result** (Kunsh) - show `fig_models_test.png`
"Hybrid scores 0.0267 against 0.0059 for best-sellers: about 4.5 times better. Hit rate at 10: 8.3% against 2.7%. Every model except SVD beats best-sellers, with confidence intervals far from zero. The numbers look small because each shopper picks from 12,000 products and has one or two later purchases."

**Slide 5: Live demo** (Rishit)
Open the UI. Click "Show another shopper": point at the green ticks ("things they really bought afterwards, which the model never saw"). Show one miss honestly. Then the "Similar products" tab with "mascara".

**Slide 6: New products and small data** (Kunsh) - show `fig_e3_training_size_beauty.png`
"For products with no history, collaborative models find 0%; the hybrid finds 8% because it reads the description. And with only a quarter of the data, collaborative models fall below best-sellers: they need history."

**Slide 7: Honest findings and limits** (Rishit)
"SVD, which predicts star ratings, improved rating error but ranked worse than best-sellers: guessing ratings and ranking products are different jobs. Our first dataset, a 1M-row Kaggle file, behaved like random purchases and nothing beat best-sellers; we proved that with four tests and switched to a real dataset. Limits: only the Beauty category, a pre-filtered 5-core benchmark, data ends in 2014, offline evaluation, one random seed."

**Slide 8: Takeaway** (Kunsh)
"Evaluate honestly: split by time, freeze settings, test once, report failures. Under that protocol a hybrid of purchase history and product text beat best-sellers by 4.5 times, and was the only method that handled new products."

## Likely questions (short answers)
- Only beauty? Yes; the code is category-agnostic, and Amazon publishes other categories in the same format.
- Why not neural models? Data-hungry and out of time; listed as future work.
- Did you peek at the test? No: settings were committed in `manifests/frozen_settings.json` before the single test run.
- Why is 0.0267 good? It is 4.5x the baseline with a significant interval; absolute values are small for 12,000 items.
