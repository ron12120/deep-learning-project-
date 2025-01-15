import pandas as pd
import requests
import time

# reading the Excel file
file_path = './Fifa_23_Players_Data_with_Wikipedia.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1').head(1701)

# Define worth categories based on ranges
def categorize_worth(value):
    if value > 40_000_000:  # Greater than €40M
        return "Worth a lot"
    elif 10_000_000 <= value <= 40_000_000:  # Between €10M and €40M
        return "Worth above average"
    elif 1_000_000 <= value < 10_000_000:  # Between €1M and €10M
        return "Worth average"
    else:  # Less than €1M
        return "Worth low"

# Check if 'Value(in Euro)' and 'Wikipedia_Intro' columns exist
if 'Value(in Euro)' in df.columns and 'Wikipedia_Intro' in df.columns:
    # Add worth description to the 'Wikipedia_Intro' column
    df['Wikipedia_Intro'] = df.apply(
        lambda row: f"{row['Wikipedia_Intro']}  {categorize_worth(row['Value(in Euro)'])}",
        axis=1
    )

    # Save the updated DataFrame to a new file
    output_path = './Fifa_23_Players_Data_with_Wikipedia_and_Descriptions.csv'
    df.to_csv(file_path, index=False, encoding='ISO-8859-1')
    print(f"The file with updated Wikipedia intros has been saved to {file_path}")

else:
    print("Required columns ('Value(in Euro)' and/or 'Wikipedia_Intro') are missing")

# # 'Full Name' is the column of the names of the players
# if 'Full Name' in df.columns:
#     def get_wikipedia_intro(player_name):
#         try:
#             # Use Wikipedia API to fetch the summary
#             api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{player_name}"
#             headers = {'User-Agent': 'Mozilla/5.0'}
#             response = requests.get(api_url, headers=headers, timeout=10)
#             response.raise_for_status()
#             data = response.json()
#             return data.get('extract', 'No content found')
#
#         except requests.exceptions.Timeout:
#             return "Request timed out"
#
#         except requests.exceptions.HTTPError as e:
#             return f"HTTP error: {e.response.status_code}"
#
#         except Exception as e:
#             return f"Error: {e}"
#
#     # creating the new column for the player wikipedia intro
#     df['Wikipedia_Intro'] = None  # Initialize the new column with None (or NaN)
#
#     # Fetch Wikipedia intros for players between 1000 and 2000
#     for i in range(970, min(2000, len(df))):
#         player_name = df.at[i, 'Full Name']
#         intro = get_wikipedia_intro(player_name)
#
#         if intro == "HTTP error: 404":
#             player_name = df.at[i, 'Known As']
#             intro = get_wikipedia_intro(player_name)
#
#         print(f"Intro for {player_name}: {intro}")
#         df.at[i, 'Wikipedia_Intro'] = intro
#         time.sleep(1)  # 1-second delay between requests
#
#     # saving the new file
#     output_path = './Fifa_23_Players_Data_with_Wikipedia.csv'
#     df.to_csv(output_path, index=False)
#     print(f"The file has been saved to {output_path}")
#
# else:
#     print("'Known As' column not found")
