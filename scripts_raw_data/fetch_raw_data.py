import os
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from io import StringIO

# If you encounter the error:
# [ERROR] Failed for season XXXX: 429 Client Error: Too Many Requests for url
# it means Basketball Reference is temporarily blocking your IP due to too many requests.
# To resolve this, you can either:
#  - switch to another internet connection (e.g., use a phone hotspot),
#  - or disconnect and reconnect your internet to get a new IP.
# After that, re-run the script and it should work again.


def download_season_stats(start_year: int=1956, end_year: int=2025,
                          save_dir: str="raw_player_csv") -> None:
    """
    Scrapes per-season player statistics from Basketball Reference and saves them as CSV files.

    Parameters:
        start_year (int): First season to fetch (inclusive).
        end_year (int): Last season to fetch (inclusive).
        save_dir (str): Directory where the CSV files will be stored.

    Returns:
        None
    """
    print()
    print(f"[INFO] Downloading per-season raw data")
    save_dir = os.path.join(raw_data_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        print(f"[INFO] Downloading season {year} from {url}...")

        output_path = os.path.join(save_dir, f"NBA_{year}_per_game.csv")
        if os.path.exists(output_path):
            print(f"[SKIP] Skipping season {year}, file already exists.")
            continue    

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse the HTML table using pandas
            dfs = pd.read_html(response.text)
            df = dfs[0]

            # Remove header rows duplicated in the table
            df = df[df["Player"] != "Player"]
            df["Season"] = year

            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"[OK] Saved to {output_path}")

        except Exception as e:
            print(f"[WARN] Failed for season {year}: {e}, skipping")

    print(f"[DONE] Saved raw data to {save_dir}")


def download_expanded_standings(start_year: int=1956, end_year: int=2025, 
                                save_dir: str="raw_standings_csv") -> None:
    """
    Downloads the expanded team standings table for each NBA season and saves them as CSV files.

    Parameters:
        start_year (int): First season to fetch (inclusive).
        end_year (int): Last season to fetch (inclusive).
        save_dir (str): Directory where the CSV files will be stored.

    Returns:
        None
    """
    print()
    print(f"[INFO] Downloading expanded standings for each season")
    save_dir = os.path.join(raw_data_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        print(f"[INFO] Downloading expanded standings for {year}...")
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"

        output_path = os.path.join(save_dir, f"NBA_{year}_expanded_standings.csv")
        if os.path.exists(output_path):
            print(f"[SKIP] Skipping {year}, already exists.")
            continue

        try:
            res = requests.get(url)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")

            # First try to find it directly
            table = soup.find("table", id="expanded_standings")
            if table:
                df = pd.read_html(StringIO(str(table)), header=1)[0]
                df["Season"] = year
                df.to_csv(output_path, index=False)
                print(f"[OK] Found directly and saved for {year}")
                continue

            # Fallback to commented tables
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                try:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    table = comment_soup.find("table", id="expanded_standings")
                    if table:
                        df = pd.read_html(StringIO(str(table)), header=1)[0]
                        df["Season"] = year
                        df.to_csv(output_path, index=False)
                        print(f"[OK] Found in comment and saved for {year}")
                        break
                except Exception:
                    continue

            if not os.path.exists(output_path):
                print(f"[WARN] Expanded Standings not found for {year}, skipping")

        except Exception as e:
            print(f"[WARN] Failed for season {year}: {e}, skipping")

    print(f"[DONE] Saved raw data to {save_dir}")


if __name__ == "__main__":
    # Example usage:
    # python scripts_data_process/download_raw_data.py 
    # python scripts_data_process/download_raw_data.py --start 1957
    # python scripts_data_process/download_raw_data.py --start 1980 --end 2020
    
    # Constants
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_dir = os.path.join(base_dir, "raw_data")

    MIN_YEAR = 1956
    MAX_YEAR = 2025

    # Argument parser
    parser = argparse.ArgumentParser(description="Download NBA raw data from Basketball Reference.")

    parser.add_argument("--start", type=int, default=MIN_YEAR,
                        help=f"Start year (default {MIN_YEAR})")
    parser.add_argument("--end", type=int, default=MAX_YEAR,
                        help=f"End year (default {MAX_YEAR})")

    args = parser.parse_args()

    # Sanity checks
    year_start = args.start
    year_end = args.end

    if year_start < MIN_YEAR:
        print(f"[ERROR] start {year_start} is below minimum allowed ({MIN_YEAR})")
        sys.exit(1)
    if year_end > MAX_YEAR:
        print(f"[ERROR] end {year_end} is above maximum allowed ({MAX_YEAR})")
        sys.exit(1)
    if year_start > year_end:
        print(f"[ERROR] start {year_start} must be <= end {year_end}")
        sys.exit(1)

    # Run
    print()
    print(f"[INFO] Downloading '{args.what}' from {year_start} to {year_end}...\n")

    download_season_stats(year_start, year_end)
    download_expanded_standings(year_start, year_end)

    print()
    print("[INFO] All requested downloads finished.")
