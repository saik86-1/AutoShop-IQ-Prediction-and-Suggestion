import pandas as pd

def analyze_make_mileage_summary(csv_path):
    df = pd.read_csv(csv_path)

    # Convert types
    df["mileage_in"] = pd.to_numeric(df["mileage_in"], errors="coerce")
    df["labor_total_cost"] = pd.to_numeric(df["labor_total_cost"], errors="coerce")
    df["part_total_cost"] = pd.to_numeric(df["part_total_cost"], errors="coerce")

    df = df[df["mileage_in"] >= 0]  # Filter out negative mileage

    # Add mileage bucket
    bins = [0, 15000, 30000, 60000, 90000, 120000, 150000, 200000, df["mileage_in"].max()]
    labels = ["0-15k", "15-30k", "30-60k", "60-90k", "90-120k", "120-150k", "150-200k", "200k+"]
    df["mileage_bucket"] = pd.cut(df["mileage_in"], bins=bins, labels=labels, include_lowest=True)

    # Total cost
    df["total_cost"] = df["labor_total_cost"].fillna(0) + df["part_total_cost"].fillna(0)

    # Grouped summary
    grouped = df.groupby(["make", "mileage_bucket"]).agg(
        avg_labor_cost=("labor_total_cost", "mean"),
        avg_part_cost=("part_total_cost", "mean"),
        total_repairs=("part_name", "count"),
        total_cost_sum=("total_cost", "sum")
    ).reset_index()

    return grouped
