"""Project-wide settings. One place, so every experiment uses the same values."""
from pathlib import Path

SEED = 42                                   # fixed random seed (reproducibility)
ROOT = Path(__file__).resolve().parents[1]  # the recsys-project folder
RAW_DIR = ROOT.parent                       # CSVs currently sit one folder up (Downloads/archive)
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_DIR = ROOT / "manifests"

# Time-based split from the report: 70% train / 15% validation / 15% locked test
SPLIT = (0.70, 0.15, 0.15)
K_CORE = 2   # changed from the planned 5: only 5,094 of 603,815 users have >=5 purchases

RAW_FILES = {
    "amazon_ecommerce_1M.csv": {
        "source": "https://www.kaggle.com/datasets/sharmajicoder/amazon-e-commerce",
        "role": "collaborative models, popularity, time split (user_id, product_id, purchase_date)",
        "licence": "check Kaggle dataset page (not yet verified)",
    },
    "products.csv": {
        "source": "https://www.kaggle.com/datasets/lazylad99/amazon-e-commerce-product-and-review-dataset",
        "role": "content-based TF-IDF model (titles, descriptions, categories)",
        "licence": "MIT (per Kaggle page)",
    },
    "reviews.csv": {
        "source": "https://www.kaggle.com/datasets/lazylad99/amazon-e-commerce-product-and-review-dataset",
        "role": "review text and sentiment; NO user IDs, so not usable for collaborative filtering",
        "licence": "MIT (per Kaggle page)",
    },
}
