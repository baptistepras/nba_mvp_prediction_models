import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from sklearn.impute import KNNImputer

# This pipeline (1) constructs per-season player datasets,
# collecting all basic player statistics available since 1956
# (e.g., G / MP / FG / FGA / FG% / FT / FTA / FT% / TRB / AST / PF / PTS),
# and enriching them with team context: team rank, win-loss differential, and one-hot position encoding.
# It prepares aligned inputs and labels (top-1, top-10 MVP) for downstream MVP prediction tasks.
# It builds Data.csv (features per player), Y_top1.csv (binary label: MVP or not), 
# Y_top10.csv (MVP-rank or -1), and Name.csv (player names), all aligned row-by-row.


def parse_team_overall(record: str) -> float:
    """
    Parses a team's overall record string of the form 'W–L' and returns the win-loss differential.

    Parameters:
        overall_str (str): The team's overall record string, e.g., "45-27".

    Returns:
        float: Win-loss differential (W - L), or NaN if the input is invalid.
    """
    try:
        wins, losses = map(int, record.split("-")[:2])
        return wins - losses
    except:
        return np.nan


def one_hot_pos(pos: str) -> List[int]:
    """
    Encodes a player's position as a one-hot vector over the specified position classes.

    Parameters:
        pos (str): The player's position string (e.g., 'SF').

    Returns:
        list[int]: One-hot encoded vector of length len(POS_ORDER), or all zeros if pos is unknown or missing.
    """
    if not isinstance(pos, str):
        return [0] * len(POS_ORDER)
    vec = [1 if p == pos else 0 for p in POS_ORDER]
    return vec if any(vec) else [0] * len(POS_ORDER)


def create_pipeline_allStats_from1956(start_year: int=1956, end_year: int=2025,
                                   raw_data_dir: str="raw_data",
                                   output_dir: str="allStats_from1956") -> None:
    """
    Generates processed CSVs for each season in the format specified by pipeline 1.

    Parameters:
        start_year (int): Start season (inclusive).
        end_year (int): End season (inclusive).
        raw_data_dir (str): Path to raw data.
        output_dir (str): Path to output directory.

    Returns:
        None
    """
    print()
    print(f"[INFO] Processing the raw data to per-season cleaned data")
    player_dir = os.path.join(raw_data_dir, "raw_player_csv")
    standings_dir = os.path.join(raw_data_dir, "raw_standings_csv")
    mapping_path = os.path.join(raw_data_dir, "team_id_to_name.csv")
    output_dir = os.path.join(processed_dir, output_dir)

    team_mapping = pd.read_csv(mapping_path).set_index("Team ID")["Full Name"].to_dict()

    os.makedirs(output_dir, exist_ok=True)

    for year in tqdm(range(start_year, end_year + 1), desc="Creating per-season cleaned data", file=sys.stdout):
        try:
            player_file = os.path.join(player_dir, f"NBA_{year}_per_game.csv")
            standings_file = os.path.join(standings_dir, f"NBA_{year}_expanded_standings.csv")
            if not os.path.isfile(player_file):
                print(f"[WARN] Player file missing for {year}: {player_file}, skipping")
                continue
            if not os.path.isfile(standings_file):
                print(f"[WARN] Standings file missing for {year}: {standings_file}, skipping")
                continue
            df_players = pd.read_csv(player_file)
            df_standings = pd.read_csv(standings_file)

            standings_lookup = df_standings.set_index("Team")

            data_rows = []
            y_top1 = []
            y_top10 = []
            names = []
            seen_players = set()

            previous_team_id = None
            previous_player = None

            for idx, row in df_players.iterrows():
                player = row["Player"]
                team_id = row["Team"]
                awards = row.get("Awards", "")

                if player == "League Average":
                    continue
                if player == previous_player:
                    # If the current row belongs to the same player as the previous one, and the previous row 
                    # had a generic team ID like "2TM", we retroactively assign the specific team rank and 
                    # overall to that previous summary row using this new, specific team info.
                    # This ensures the summary row isn't left with missing team information.
                    full_team = team_mapping.get(team_id, None)
                    if full_team is not None and full_team in standings_lookup.index:
                        row_team = standings_lookup.loc[full_team]
                        team_rank = row_team["Rk"] if "Rk" in row_team else np.nan
                        team_overall = parse_team_overall(row_team["Overall"] if "Overall" in row_team else "")
                    else:
                        team_rank = np.nan
                        team_overall = np.nan

                    if len(data_rows) > 0:
                        data_rows[-1]["Team Rank"] = team_rank
                        data_rows[-1]["Team Overall"] = team_overall

                    previous_player = None
                    continue 
                if player in seen_players:
                    continue
                seen_players.add(player)

                # MVP labels
                if isinstance(awards, str) and "MVP" in awards:
                    rank = awards.split("MVP-")[-1].split(",")[0] if "-" in awards else "1"
                    y_top1.append(1 if rank == "1" else 0)
                    y_top10.append(int(rank))
                else:
                    y_top1.append(0)
                    y_top10.append(-1)

                # Name
                names.append(player)

                # Team mapping handling
                if team_id in ["2TM", "3TM", "4TM", "5TM"]:
                    team_id = previous_team_id if previous_player == player else None
                elif team_id == "CHA" and year <= 2014:
                    team_id = "CHO"
                elif team_id == "CHO" and year > 2014:
                    team_id = "CHA"

                previous_player = player
                previous_team_id = team_id

                full_team = team_mapping.get(team_id, None)

                if full_team is not None and full_team in standings_lookup.index:
                    row_team = standings_lookup.loc[full_team]
                    team_rank = row_team["Rk"] if "Rk" in row_team else np.nan
                    team_overall = parse_team_overall(row_team["Overall"] if "Overall" in row_team else "")
                else:
                    team_rank = np.nan
                    team_overall = np.nan

                stats = {
                    "Team Rank": team_rank,
                    "Team Overall": team_overall,
                    "G": row.get("G", np.nan),
                    "MP": row.get("MP", np.nan),
                    "FG": row.get("FG", np.nan),
                    "FGA": row.get("FGA", np.nan),
                    "FG%": row.get("FG%", np.nan),
                    "FT": row.get("FT", np.nan),
                    "FTA": row.get("FTA", np.nan),
                    "FT%": row.get("FT%", np.nan),
                    "TRB": row.get("TRB", np.nan),
                    "AST": row.get("AST", np.nan),
                    "PF": row.get("PF", np.nan),
                    "PTS": row.get("PTS", np.nan),
                }

                pos_vector = one_hot_pos(row.get("Pos", None))
                for i, tag in enumerate(POS_ORDER):
                    stats[f"POS_{tag}"] = pos_vector[i]

                data_rows.append(stats)

            year_dir = os.path.join(output_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)

            pd.DataFrame(data_rows).to_csv(os.path.join(year_dir, "Data.csv"), index=False)
            pd.Series(y_top1, name="Top1").to_csv(os.path.join(year_dir, "Y_top1.csv"), index=False)
            pd.Series(y_top10, name="Top10").to_csv(os.path.join(year_dir, "Y_top10.csv"), index=False)
            pd.Series(names, name="Name").to_csv(os.path.join(year_dir, "Name.csv"), index=False)

        except Exception as e:
            print(f"[WARN] Failed for year {year}: {e}, skipping")

    print(f"[DONE] Saved per-season data to {output_dir}")


def restore_one_hot(df: pd.DataFrame, pos_columns: List[str]) -> pd.DataFrame:
    """
    Restores one-hot encoding by setting the max value to 1 and the others to 0 across POS_* columns.

    Parameters:
        df (pd.DataFrame): DataFrame with float values in POS_* columns.
        pos_columns (list[str]): List of POS_* column names.

    Returns:
        pd.DataFrame: DataFrame with restored one-hot encoding.
    """
    pos_data = df[pos_columns].values
    restored = np.zeros_like(pos_data)
    for i, row in enumerate(pos_data):
        if np.allclose(row, 0):
            continue 
        max_idx = np.argmax(row)
        restored[i, max_idx] = 1

    df[pos_columns] = restored.astype(int)
    return df


def knn_impute_data_csv(start_year: int=1956, end_year: int=2025, k: int=10,
                         data_dir: str="allStats_from1956") -> None:
    """
    Applies KNN imputation to fill missing values in Data.csv files for each year.

    Parameters:
        start_year (int): First season to impute (inclusive).
        end_year (int): Last season to impute (inclusive).
        k (int): Number of neighbors to use for imputation.
        data_dir (str): Path to the directory containing the yearly subdirectories with Data.csv.

    Returns:
        None
    """
    print()
    print(f"[INFO] Filling missing values for per-season data")
    data_dir = os.path.join(processed_dir, data_dir)
    pos_columns = [f"POS_{pos}" for pos in POS_ORDER]
    imputer = KNNImputer(n_neighbors=k)

    has_error = False

    for year in tqdm(range(start_year, end_year + 1), desc="Imputing missing values with KNN", file=sys.stdout):
        try:
            data_path = os.path.join(data_dir, str(year), "Data.csv")
            if not os.path.isfile(data_path):
                print(f"[WARN] File missing: {data_path}, skipping")
                continue
            df = pd.read_csv(data_path)

            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df_imputed = restore_one_hot(df_imputed, pos_columns)
            df_imputed.to_csv(data_path, index=False)
        except Exception as e:
            print(f"[ERROR] Failed KNN imputation for {year}: {e}")
            has_error = True

    if not has_error:
        print(f"[DONE] Finished filling missing values.")


def normalize_data_csv(start_year: int=1956, end_year: int=2025,
                        data_dir: str="allStats_from1956") -> None:
    """
    Normalizes the Data.csv file for each season in a given range.
    Applies Z-score normalization to all features except Team Rank, which is normalized as rank / nb_teams.

    Parameters:
        start_year (int): First season to normalize (inclusive).
        end_year (int): Last season to normalize (inclusive).
        data_dir (str): Path to the directory containing the yearly subdirectories with Data.csv.

    Returns:
        None
    """
    print()
    print(f"[INFO] Normalizing per-season data")
    data_dir = os.path.join(processed_dir, data_dir)

    has_error = False

    for year in tqdm(range(start_year, end_year + 1), desc="Normalizing cleaned stats", file=sys.stdout):
        try:
            data_path = os.path.join(data_dir, str(year), "Data.csv")
            if not os.path.isfile(data_path):
                print(f"[WARN] File missing: {data_path}, skipping")
                continue
            df = pd.read_csv(data_path)

            # Normalize Team Rank
            if "Team Rank" in df.columns:
                nb_teams = df["Team Rank"].nunique()
                df["Team Rank"] = df["Team Rank"] / nb_teams

            # Z-score normalization for all other numeric columns (excluding team rank and one-hot)
            for col in df.columns:
                if col not in ["Team Rank", "POS_C", "POS_PG", "POS_PF", "POS_SF", "POS_SG"]:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std

            df.to_csv(data_path, index=False)
        except Exception as e:
            print(f"[ERROR] Failed normalization for {year}: {e}")
            has_error = True

    if not has_error:
        print(f"[DONE] Finished normalization.")


def check_processed_data_integrity(start_year: int=1956, end_year: int=2025,
                                   data_dir: str="allStats_from1956") -> None:
    """
    Verifies the integrity of the processed Data.csv files for each season.
    Checks include absence of NaN values, validity of team rank normalization, 
    and correctness of one-hot encoding for positions.

    Parameters:
        start_year (int): First season to check (inclusive).
        end_year (int): Last season to check (inclusive).
        data_dir (str): Path to the directory containing the yearly subdirectories with Data.csv.

    Returns:
        None
    """
    print()
    print(f"[INFO] Verifying processed data integrity")
    data_dir = os.path.join(processed_dir, data_dir)
    pos_columns = [f"POS_{pos}" for pos in POS_ORDER]

    has_error = False

    for year in tqdm(range(start_year, end_year + 1), desc="Validating per-season files", file=sys.stdout):
        data_path = os.path.join(data_dir, str(year), "Data.csv")
        if not os.path.isfile(data_path):
            print(f"[WARN] File missing: {data_path}, skipping")
            continue
        df = pd.read_csv(data_path)

        # Check for NaN
        if df.isna().any().any():
            print(f"[ERROR] NaN values found in {year}")
            has_error = True

        # Check that Team Rank is between 0 and 1
        if "Team Rank" in df.columns:
            if not ((df["Team Rank"] >= 0) & (df["Team Rank"] <= 1)).all():
                print(f"[ERROR] Invalid Team Rank values in {year} (expected in [0, 1])")
                has_error = True

        # Check that position one-hot encoding is 0 or 1
        for col in pos_columns:
            if not df[col].isin([0, 1]).all():
                print(f"[ERROR] Invalid values in column {col} for year {year} (should be 0 or 1)")
                has_error = True

        # Check that one-hot vectors sum to either 0 (unknown) or 1 (valid)
        pos_sums = df[pos_columns].sum(axis=1)
        if not pos_sums.isin([0, 1]).all():
            print(f"[ERROR] Invalid one-hot encoding in year {year} (row sums should be 0 or 1)")
            has_error = True

    if not has_error:
        print("[DONE] Finished checking processed data integrity.")


if __name__ == "__main__":
    # Example usage from root:
    # python scripts_data_process/build_allStats_from1956.py
    # python scripts_data_process/build_allStats_from1956.py --start 1980 --end 2020
    
    # Set constants
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_dir = os.path.join(base_dir, "raw_data")
    processed_dir = os.path.join(base_dir, "processed_data")
    POS_ORDER = ["C", "PG", "PF", "SF", "SG"]

    # Set min/max allowed years for this pipeline
    MIN_YEAR = 1956
    MAX_YEAR = 2025

    # Setup argparse
    parser = argparse.ArgumentParser(description="Build and process the allStats_from1956 pipeline.")
    parser.add_argument("--start", type=int, default=MIN_YEAR, help="Start year (default: {MIN_YEAR})")
    parser.add_argument("--end", type=int, default=MAX_YEAR, help="End year (default: {MAX_YEAR})")

    args = parser.parse_args()
    year_start = args.start
    year_end = args.end

    # Sanity checks
    if year_start < MIN_YEAR:
        print(f"[ERROR] start {year_start} is below minimum allowed ({MIN_YEAR})")
        sys.exit(1)
    if year_end > MAX_YEAR:
        print(f"[ERROR] end {year_end} is above maximum allowed ({MAX_YEAR})")
        sys.exit(1)
    if year_start > year_end:
        print(f"[ERROR] start {year_start} must be <= end-year {year_end}")
        sys.exit(1)

    print(f"[INFO] Running pipeline from {year_start} to {year_end}...")

    # Run pipeline steps
    create_pipeline_allStats_from1956(year_start, year_end)
    knn_impute_data_csv(year_start, year_end)
    normalize_data_csv(year_start, year_end)
    check_processed_data_integrity(year_start, year_end)
