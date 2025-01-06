import pandas as pd
import requests
import time

# reading the Excel file
file_path = './Fifa 23 Players Data.csv'
df = pd.read_csv(file_path)

# 'Full Name' is the column of the names of the players
if 'Full Name' in df.columns:
    def get_wikipedia_intro(player_name):
        try:
            # Use Wikipedia API to fetch the summary
            api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{player_name}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('extract', 'No content found')

        except requests.exceptions.Timeout:
            return "Request timed out"

        except requests.exceptions.HTTPError as e:
            return f"HTTP error: {e.response.status_code}"

        except Exception as e:
            return f"Error: {e}"

    # creating the new column for the player wikipedia intro
    df['Wikipedia_Intro'] = None  # Initialize the new column with None (or NaN)
    for i in range(min(1000, len(df))):
        player_name = df.at[i, 'Full Name']
        intro = get_wikipedia_intro(player_name)
        print(f"Intro for {player_name}: {intro}")
        df.at[i, 'Wikipedia_Intro'] = intro
        time.sleep(1)  # 1-second delay between requests

    # saving the new file
    output_path = './Fifa_23_Players_with_Wikipedia.csv'
    df.to_csv(output_path, index=False)
    print(f"the file saved in  {output_path}")

else:
    print("'Full Name' column didn't found")
