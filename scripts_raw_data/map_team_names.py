import os
import sys
import pandas as pd
import requests
import math
from bs4 import BeautifulSoup
from io import StringIO
from tqdm import tqdm
from typing import Set, Dict


def build_team_abbreviation_mapping(data_dir: str="raw_player_csv", 
                                    output_csv: str="team_id_to_name.csv") -> None:
    """
    Builds a mapping between team abbreviations (as seen in player CSVs) and full team names
    by scraping Basketball Reference team index page. Only team abbreviations found in the
    provided CSVs are retained.

    Parameters:
        data_dir (str): Path to directory containing player stats CSVs.
        output_csv (str): Path to save the resultxing abbreviation -> full name mapping.

    Returns:
        None
    """
    print()
    print(f"[INFO] Building team abbreviation mapping")
    data_dir = os.path.join(raw_data_dir, data_dir)
    output_csv = os.path.join(raw_data_dir, output_csv)
    team_ids: Set[str] = set()

    # Collect all team abbreviations present in player CSVs
    for file in tqdm(os.listdir(data_dir), desc="Scanning player CSVs for team IDs", file=sys.stdout):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file))
            if "Team" in df.columns:
                team_ids.update(df["Team"].dropna().unique())

    # Scrape Basketball Reference for all team abbreviation ↔ name mappings
    url = "https://www.basketball-reference.com/teams/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    mapping: Dict[str, str] = {}
    for table_id in ["teams_active", "teams_defunct"]:
        table = soup.find("table", {"id": table_id})
        if table:
            for row in table.tbody.find_all("tr"):
                link = row.find("a")
                if link and "/teams/" in link["href"]:
                    abbr = link["href"].split("/")[2]
                    name = link.text.strip()
                    if abbr in mapping and mapping[abbr] != name:
                        print(f"[WARN] Warning: Team abbreviation conflict for {abbr}: '{mapping[abbr]}' vs '{name}'")
                        print(f"[ERROR] Run merge_team_mapping() to resolve them.")
                    mapping[abbr] = name

    # Filter only team ids seen in player stats
    filtered_mapping = {abbr: mapping[abbr] for abbr in team_ids if abbr in mapping}

    # Report unmapped abbreviations
    unknown = team_ids - set(filtered_mapping.keys())
    if unknown:
        print(f"[WARN] Unknown team abbreviations not found on site: {sorted(unknown)}")
        print(f"[ERROR] Run merge_team_mapping() to resolve them.")
    # Save to CSV
    df_out = pd.DataFrame(list(filtered_mapping.items()), columns=["Team ID", "Full Name"])
    df_out.to_csv(output_csv, index=False)

    print(f"[DONE] Saved mapping to {output_csv}")


def merge_team_mapping(auto_mapping_path: str="team_id_to_name.csv", 
                       manual_mapping_path: str="historical_teams.csv",
                       out_path: str="team_id_to_name.csv") -> None:
    """
    Merges automatic and manual team abbreviation-to-name mappings into a single CSV file.

    Parameters:
        auto_mapping_path (str): Path to the automatically generated team mapping CSV.
        manual_mapping_path (str): Path to the manually curated team mapping CSV.
        out_path (str): Path where the merged mapping CSV will be saved.

    Returns:
        None
    """
    print()
    print(f"[INFO] Fetching missing teams from {manual_mapping_path}")
    auto_mapping_path = os.path.join(raw_data_dir, auto_mapping_path)
    manual_mapping_path = os.path.join(raw_data_dir, manual_mapping_path)
    out_path = os.path.join(raw_data_dir, out_path)

    auto_df = pd.read_csv(auto_mapping_path)
    manual_df = pd.read_csv(manual_mapping_path)

    merged_df = pd.concat([auto_df, manual_df]).drop_duplicates(subset=["Team ID"]).reset_index(drop=True)
    
    # Manual corrections
    corrections = {
        "NJN": "New Jersey Nets",
        "NOH": "New Orleans Hornets",
        "KCO": "Kansas City-Omaha Kings",
        "CHH": "Charlotte Hornets"
    }
    for team_id, correct_name in corrections.items():
        if team_id in merged_df["Team ID"].values:
            merged_df.loc[merged_df["Team ID"] == team_id, "Full Name"] = correct_name

    merged_df.to_csv(out_path, index=False)
    print(f"[DONE] Merged mapping saved to {out_path}")


def check_missing_teams(stats_dir: str="raw_player_csv",
                        standings_dir: str="raw_standings_csv",
                        mapping_path: str="team_id_to_name.csv") -> None:
    """
    Checks if there are any team abbreviations in the raw stats CSVs that are missing from the mapping.

    Parameters:
        stats_dir (str): Directory containing raw player CSVs.
        standings_dir (str): Directory containing raw expanded standings CSVs.
        mapping_path (str): Path to the team mapping CSV.

    Returns:
        None
    """
    print()
    print(f"[INFO] Checking for missing teams")
    stats_dir = os.path.join(raw_data_dir, stats_dir)
    standings_dir = os.path.join(raw_data_dir, standings_dir)
    mapping_path = os.path.join(raw_data_dir, mapping_path)

    # Load mapping
    mapping_df = pd.read_csv(mapping_path)
    mapped_abbrs = set(mapping_df["Team ID"].dropna().astype(str).unique())
    mapped_names = set(mapping_df["Full Name"].dropna().astype(str).unique())

    # Check team abbreviations in player stats
    abbreviations_found = set()
    for file in tqdm(os.listdir(stats_dir), desc="Checking player team IDs", file=sys.stdout):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(stats_dir, file))
            team_column = "Team" if "Team" in df.columns else None
            if team_column:
                abbreviations_found.update(map(str, df[team_column].dropna().unique()))

    missing_abbrs = abbreviations_found - mapped_abbrs - {'2TM', '3TM', '4TM', '5TM'}
    if missing_abbrs:
        print(f"[WARN] Missing team abbreviations in mapping: {sorted(missing_abbrs)}")
        print(f"[ERROR] You must resolve these missing teams manually mappings before continuing.")
    else:
        print("[DONE] All team abbreviations in player stats are mapped.")

    # Check full team names in standings
    fullnames_found = set()
    for file in tqdm(os.listdir(standings_dir), desc="Checking standings team names", file=sys.stdout):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(standings_dir, file))
            team_column = "Team" if "Team" in df.columns else None
            if team_column:
                fullnames_found.update(map(str, df[team_column].dropna().unique()))

    missing_names = fullnames_found - mapped_names
    if missing_names:
        print(f"[WARN] Missing full team names in mapping: {sorted(missing_names)}")
        print(f"[ERROR] You must resolve these missing teams manually mappings before continuing.")
    else:
        print("[DONE] All full team names in standings are mapped.")


def check_duplicate_team_ids(mapping_path: str="team_id_to_name.csv") -> None:
    """
    Checks for duplicate Team ID entries in the mapping file.

    Parameters:
        mapping_path (str): Path to the team ID to name mapping CSV.

    Returns:
        None
    """
    print()
    print(f"[INFO] Checking for duplicate team IDs")
    mapping_path = os.path.join(raw_data_dir, mapping_path)

    df = pd.read_csv(mapping_path)
    duplicated = df[df["Team ID"].duplicated(keep=False)]

    if duplicated.empty:
        print("[DONE] No duplicate Team ID found in the mapping.")
    else:
        print("[WARN] Duplicate Team ID(s) found:")
        print(duplicated.sort_values("Team ID").to_string(index=False))
        print(f"[ERROR] You must resolve these duplicates in the mapping manually before continuing.")


def check_player_to_standings_mapping(stats_dir: str="raw_player_csv",
                                      standings_dir: str="raw_standings_csv",
                                      mapping_path: str="team_id_to_name.csv") -> None:
    """
    For each season, checks that all team abbreviations used in player stats CSVs
    can be mapped via the team mapping file and matched with a team in the standings file.

    Parameters:
        stats_dir (str): Directory containing raw player stats CSVs.
        standings_dir (str): Directory containing raw standings CSVs.
        mapping_path (str): Path to the team ID to name mapping CSV.

    Returns:
        None
    """
    print()
    print(f"[INFO] Checking mapping")
    stats_dir = os.path.join(raw_data_dir, stats_dir)
    standings_dir = os.path.join(raw_data_dir, standings_dir)
    mapping_path = os.path.join(raw_data_dir, mapping_path)

    # Load team abbreviation -> full name mapping
    mapping_df = pd.read_csv(mapping_path)
    team_id_to_name = dict(zip(mapping_df["Team ID"], mapping_df["Full Name"]))

    has_error = False

    for file in tqdm(sorted(os.listdir(stats_dir)), desc="Validating player-to-standings mapping", file=sys.stdout):
        if not file.endswith(".csv"):
            continue
        year = file.split("_")[1]
        player_path = os.path.join(stats_dir, file)
        standings_path = os.path.join(standings_dir, f"NBA_{year}_expanded_standings.csv")

        if not os.path.exists(standings_path):
            print(f"[WARN] Standings file missing for {year}, skipping")
            continue

        df_players = pd.read_csv(player_path)
        df_standings = pd.read_csv(standings_path)
        
        player_team_ids = set(df_players["Team"].unique())
        standings_team_names = set(df_standings["Team"].unique())

        for team_id in player_team_ids:
            if team_id == "CHA" and int(year) <= 2014:
                team_id = "CHO"  # Charlotte Bobcats
            elif team_id == "CHO" and int(year) > 2014:
                team_id = "CHA"  # Charlotte Hornets
            if team_id not in team_id_to_name or pd.isna(team_id):
                if team_id in ["2TM", "3TM", "4TM", "5TM"] or isinstance(team_id, float) and math.isnan(team_id):
                    continue
                else:
                    print(f"[WARN] {team_id} (from {year}) missing in mapping")
                    continue
            full_name = team_id_to_name[team_id]
            if full_name not in standings_team_names:
                print(f"[WARN] {team_id} -> '{full_name}' not found in standings {year}")
                has_error = True

    if not has_error:
        print("[DONE] Finished checking player-to-standings mapping.")
    else:
        print(f"[ERROR] You must resolve the missing team mappings manually before continuing.")


if __name__ == "__main__":
    # Example usage from root:
    # python scripts_data_process/build_team_mapping.py

    # Constants
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_dir = os.path.join(base_dir, "raw_data")
    
    build_team_abbreviation_mapping()
    merge_team_mapping()
    check_missing_teams()
    check_duplicate_team_ids()
    check_player_to_standings_mapping()
