# E-commerce Sales Analysis Project

## Project Overview
This project explores an e-commerce sales dataset to understand sales patterns, profit trends, and category performance. I started with exploratory data analysis in Jupyter Notebook, then improved the project structure, documentation, and Git workflow through branch-based development.

## Main Goal
The goal of this project was to analyze e-commerce sales data and learn how to manage a data project properly using Git and GitHub. I wanted to understand what the dataset could show about sales, profit, categories, and regions, while also making the project more organized and easier to follow.

## Dataset
Original dataset source:
- E-commerce Orders and Sales Performance Dataset (Kaggle)
- https://www.kaggle.com/datasets/zahranusratt/e-commerce-orders-and-sales-performance-dataset

Additional dataset used later in the project:
- `Category_Details.csv`
- A synthetic lookup dataset created to join on `Category`

## What I Did

### Part 1: Exploratory Data Analysis
I explored the original e-commerce dataset by:
- checking the structure and columns
- reviewing summary statistics
- analyzing sales and profit
- comparing category and region performance
- creating visualizations to better understand patterns

### Part 2: Project Structure and Join Work
In the next stage of the project, I improved the repository structure by organizing files into folders such as:
- `scripts/`
- `data/raw/`
- `data/clean/`
- `docs/`
- `results/`

I also created a second notebook where I joined the original sales data with a category lookup dataset using a **left join** on `Category`.

## Key Findings
- Sales vary a lot across orders, with many smaller orders and some much larger ones.
- Categories contribute differently to total sales and profit.
- Regions also differ in profitability.
- The join dataset helped enrich the project by adding category-level information such as margin, return rate, and priority.
- The join also showed how unmatched keys can create `NaN` values, which is an important part of real data work.

## Tools Used
- Python
- pandas
- matplotlib
- Jupyter Notebook
- Git
- GitHub

## Project Structure
- `scripts/` → notebooks for analysis and join work
- `data/raw/` → raw input data files
- `data/clean/` → processed output files
- `docs/` → HTML/Markdown exports and screenshots
- `results/` → additional outputs
- `citations.md` → source references and dataset notes

## What I Learned
This project helped me learn not just data analysis, but also how to manage a project more professionally. I learned how useful Git branches are, why `.gitignore` matters, how relative paths make a project portable, and how documentation improves the quality of a project. I also learned that joins are not just about combining data, but also about understanding mismatches and missing values.

## Notes
- The toddler project work has now been merged into `main`.
- The final reflection for the assignment was submitted separately to Learning Hub.
- AI was used as a support tool during planning, structuring, explanations, and debugging.