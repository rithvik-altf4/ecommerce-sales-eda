# E-commerce Sales EDA

## Data source
Kaggle dataset:  
https://www.kaggle.com/datasets/zahranusratt/e-commerce-orders-and-sales-performance-dataset

## What this project is
I explored an e-commerce sales dataset to understand what drives sales and profit.  
I looked at sales and profit by category and region, and checked how sales changes over time.

## What I found (quick summary)
- Sales are not all the same size. Most orders are smaller, but a few are much bigger.
- Some categories bring in more total sales than others.
- Some regions generate more profit than others.
- Sales and profit usually move together, but not perfectly.
- Monthly sales changes over time, so some months look stronger than others.

## Project structure

- `scripts/` → notebooks for EDA and join work
- `data/raw/` → raw input files (ignored in GitHub via `.gitignore`)
- `data/clean/` → cleaned/joined output datasets
- `docs/` → exported HTML/Markdown files and chart images
- `results/` → additional output artifacts
- `citations.md` → dataset and source references

## Files currently used

- `scripts/01_eda_ecommerce.ipynb` → main EDA notebook
- `docs/DataAnalyis_eCommerce.html` → exported HTML version
- `docs/DataAnalyis_eCommerce.md` → exported Markdown version
- `docs/output_*.png` → images used by Markdown rendering

## Part 2 join dataset

For the join task, I use:
- `data/raw/Category_Details.csv` (lookup table)

Join key:
- `Category`

Expected output:
- `data/clean/ecommerce_sales_joined.csv`

## Notes

- Raw data files are intentionally not tracked on GitHub.
- `.ipynb_checkpoints/` and `data/raw/` are ignored using `.gitignore`.
- Relative paths are used in notebooks (for example `../data/raw/...`) so the project is portable for other users.
