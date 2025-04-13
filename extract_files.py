import pandas as pd

# Load the dataset
df = pd.read_csv("IMDb Top 1000.csv")

# Extract the 'Title' column, lowercase, remove duplicates
titles = df['Title'].dropna().str.lower().drop_duplicates()

# Optional: Clean the titles (remove parentheses and trailing spaces)
titles = titles.str.replace(r"\\(.*\\)", "", regex=True).str.strip()

# Save the cleaned titles to movies.txt
titles.to_csv("movies.txt", index=False, header=False)

print("Extracted and saved 1000 movie names to 'movies.txt'")
