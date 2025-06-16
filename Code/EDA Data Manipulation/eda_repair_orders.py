import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.colors as pc
import os
from analyze_make_mileage_summary import analyze_make_mileage_summary


# === CONFIG ===
CSV_PATH = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\structured_repair_data.csv"
# OUTPUT_DIR = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\EDA_Charts"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# # === CLEANING & TYPE CASTING ===
# df["labor_total_cost"] = pd.to_numeric(df.get("labor_cost", 0), errors="coerce")
# df["part_total_cost"] = pd.to_numeric(df.get("part_cost", 0), errors="coerce")
# df["labor_hours"] = pd.to_numeric(df.get("labor_hours", 0), errors="coerce")
# df["mileage_in"] = pd.to_numeric(df["mileage_in"], errors="coerce")

# # === BASIC INFO ===
# print("📊 Dataset Size:", df.shape)
# print("🧾 Columns:", df.columns.tolist())
# print("🔧 Unique Job Types:", df["job_type"].nunique())
# print("🔩 Unique Part Names:", df["part_name"].nunique())
# print("🛠️ Unique Labor Job Info:", df["labor_job_info"].nunique())

# # === 1. TOP CONCERNS ===
# plt.figure(figsize=(10, 5))
# df["concern"].value_counts().nlargest(10).plot(kind="barh", color="skyblue")
# plt.title("Top 10 Most Common Concerns")
# plt.xlabel("Count")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "top_10_concerns.png"))
# plt.close()

# # === 2. TOP JOB TYPES PER CONCERN ===
# top_job_types = df[df["source"] == "labor"]["job_type"].value_counts().nlargest(10)
# plt.figure(figsize=(10, 5))
# top_job_types.plot(kind="bar", color="lightgreen")
# plt.title("Top 10 Job Types (Labor)")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "top_10_job_types.png"))
# plt.close()

# # === 3. TOP PARTS ===
# top_parts = df[df["source"] == "part"]["part_name"].value_counts().nlargest(10)
# plt.figure(figsize=(10, 5))
# top_parts.plot(kind="bar", color="orange")
# plt.title("Top 10 Parts Used")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "top_10_parts.png"))
# plt.close()

# # === 4. LABOR & PART COST DISTRIBUTION ===
# plt.figure(figsize=(10, 5))
# sns.histplot(df["labor_total_cost"].dropna(), bins=30, color="blue", kde=True)
# plt.title("Distribution of Labor Costs")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "labor_cost_distribution.png"))
# plt.close()

# plt.figure(figsize=(10, 5))
# sns.histplot(df["part_total_cost"].dropna(), bins=30, color="red", kde=True)
# plt.title("Distribution of Part Costs")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "part_cost_distribution.png"))
# plt.close()

# # === 5. COST vs MILEAGE ===
# plt.figure(figsize=(10, 5))
# sns.scatterplot(data=df, x="mileage_in", y="labor_total_cost", alpha=0.5)
# plt.title("Labor Cost vs Mileage")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "labor_cost_vs_mileage.png"))
# plt.close()

# plt.figure(figsize=(10, 5))
# sns.scatterplot(data=df, x="mileage_in", y="part_total_cost", alpha=0.5)
# plt.title("Part Cost vs Mileage")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "part_cost_vs_mileage.png"))
# plt.close()

# print(f"✅ EDA charts saved in → {OUTPUT_DIR}")


# print(df['labor_total_cost'].dtype)            # Should be float
# print(df['labor_total_cost'].unique()[:10])    # Preview values
# print(df['labor_total_cost'].isna().sum())     # Null values
# print(df['labor_total_cost'].value_counts().head(10))  # Common values

#-------------------------------------------------------------------------
# # 1. Box Plot to Show Spread & Outliers

# plt.figure(figsize=(10, 5))
# sns.boxplot(x=df[df['source'] == 'labor']['labor_total_cost'], color="skyblue")
# plt.title("Box Plot of Labor Total Cost (Filtered)")
# plt.xlabel("Labor Total Cost ($)")
# plt.tight_layout()
# plt.show()

# # Log Scale Histogram (for highly skewed data)
# plt.figure(figsize=(10, 5))
# sns.histplot(df[df['source'] == 'labor']['labor_total_cost'], bins=50, color="purple", log_scale=(False, True))
# plt.title("Log-Scaled Distribution of Labor Costs")
# plt.xlabel("Labor Cost ($)")
# plt.ylabel("Log Count")
# plt.tight_layout()
# plt.show()

# # 3. Violin Plot (for Category vs. Cost)

# plt.figure(figsize=(12, 6))
# top_job_types = df[df['source'] == 'labor']['job_type'].value_counts().head(5).index
# sns.violinplot(
#     data=df[(df['source'] == 'labor') & (df['job_type'].isin(top_job_types))],
#     x="job_type",
#     y="labor_total_cost",
#     palette="Pastel1"
# )
# plt.title("Labor Cost by Top 5 Job Types")
# plt.ylabel("Labor Cost ($)")
# plt.tight_layout()
# plt.show()

# # 4. Cost Distribution Summary Table

# labor_summary = df[df['source'] == 'labor']['labor_total_cost'].describe()
# part_summary = df[df['source'] == 'part']['part_total_cost'].describe()

# print("🔧 Labor Cost Summary:")
# print(labor_summary)

# print("\n🔩 Part Cost Summary:")
# print(part_summary)


# 5. Categorize Cost Ranges for Heatmaps or Pie Charts

#---------------------------------------------------------------------------------------


summary_df = analyze_make_mileage_summary(CSV_PATH)

# print(summary_df.head())  # Or pass to visualization function

# Pivot the DataFrame
# heatmap_data = summary_df.pivot(index="make", columns="mileage_bucket", values="avg_labor_cost")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Set figure size
# plt.figure(figsize=(14, 10))

# # Create the heatmap
# ax = sns.heatmap(
#     heatmap_data,
#     annot=True,
#     fmt=".0f",
#     cmap="YlOrBr",
#     linewidths=0.5,
#     cbar_kws={'label': 'Avg Labor Cost'},
#     annot_kws={"fontsize": 9, "color": "black"}  # default color
# )

# # Improve label visibility
# plt.title("🔧 Avg Labor Cost by Make & Mileage", fontsize=16)
# plt.xlabel("Mileage Bucket", fontsize=12)
# plt.ylabel("Make", fontsize=12)

# # Rotate x-axis labels
# plt.xticks(rotation=45)

# # Manually recolor annotations based on background (better contrast)
# for text in ax.texts:
#     value = text.get_text()
#     if value != '' and value != 'nan':
#         try:
#             val = float(value)
#             # Change color if background is too light
#             if val < 70:
#                 text.set_color("black")
#             elif val < 160:
#                 text.set_color("dimgray")
#             else:
#                 text.set_color("white")
#         except ValueError:
#             pass  # in case of unexpected text

# plt.tight_layout()
# plt.show()

# print (heatmap_data.head())  # Or save to CSV or other formats as needed

#----------------------------------------------------------------------------------------


# # Filter summary_df for Honda and Toyota
# selected_makes = ["Honda", "Toyota"]
# filtered_df = summary_df[summary_df["make"].isin(selected_makes)]

# # Plot
# plt.figure(figsize=(12, 6))

# for make in selected_makes:
#     make_df = filtered_df[filtered_df["make"] == make]
#     plt.plot(make_df["mileage_bucket"], make_df["avg_labor_cost"], marker='o', label=f"{make} - Labor")
#     plt.plot(make_df["mileage_bucket"], make_df["avg_part_cost"], marker='s', linestyle='--', label=f"{make} - Part")

# plt.title("Labor vs Part Cost Trend by Mileage (Honda & Toyota)")
# plt.xlabel("Mileage Bucket")
# plt.ylabel("Average Cost ($)")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#--------------------------------------------------------------------------------------------------------

# 1. Get top 20 concerns overall


# ❌ Remove irrelevant concerns
# irrelevant = ["no technician concerns found.", "no customer concerns found."]
# df = df[~df["concern"].str.lower().isin([x.lower() for x in irrelevant])]

# # ✅ Top 20 Concerns by Make + Model + Year
# top_concerns = (
#     df.groupby(["make", "model", "year", "concern"])
#     .size()
#     .reset_index(name="count")
#     .sort_values("count", ascending=False)
# )

# # Aggregate across make/model/year for ranking
# top_20 = top_concerns.groupby("concern")["count"].sum().nlargest(20)

# # 📊 Plot
# plt.figure(figsize=(10, 6))
# sns.barplot(y=top_20.index, x=top_20.values, palette="viridis")
# plt.title("Top 20 Reported Concerns Across All Vehicles")
# plt.xlabel("Number of Occurrences")
# plt.ylabel("Concerns")
# plt.tight_layout()
# plt.show()

# # 2. top 10 most frequent concerns
# # === AUTHENTICATION ===
# tls.set_credentials_file(username='mrugal', api_key='74kOWfHF2yA3epZYoTbU')

# # === LOAD & CLEAN DATA ===
# df["concern"] = df["concern"].astype(str).str.lower().str.strip()
# df["job_type"] = df["job_type"].astype(str).str.lower().str.strip()

# # ❌ Remove irrelevant concerns
# irrelevant = ["no technician concerns found.", "no customer concerns found."]
# df = df[~df["concern"].isin(irrelevant)]

# # === FILTER: Top 10 Concerns ===
# top_10_concerns = df["concern"].value_counts().nlargest(10).index.tolist()
# filtered_df = df[df["concern"].isin(top_10_concerns)]

# # === Top 3 Job Types per Concern ===
# top_jobs = (
#     filtered_df.groupby(["concern", "job_type"])
#     .size()
#     .reset_index(name="count")
#     .sort_values(["concern", "count"], ascending=[True, False])
# )
# top_3 = top_jobs.groupby("concern").head(3).reset_index(drop=True)

# # === Formatting Labels ===
# top_3["concern"] = top_3["concern"].str.title()
# top_3["job_type"] = top_3["job_type"].str.title()

# concerns = top_3["concern"].unique().tolist()
# jobs = top_3["job_type"].unique().tolist()
# labels = concerns + jobs

# # === Mapping Labels to Indices ===
# concern_idx = {c: i for i, c in enumerate(concerns)}
# job_idx = {j: i + len(concerns) for i, j in enumerate(jobs)}

# # === Sankey Link Mapping ===
# sources = top_3["concern"].map(concern_idx)
# targets = top_3["job_type"].map(job_idx)
# values = top_3["count"]

# # Generate custom colors for each node
# num_nodes = len(labels)
# # Repeat colors if we have more nodes than color options
# colors = pc.qualitative.Light24 * (num_nodes // len(pc.qualitative.Light24) + 1)

# # Build the Sankey figure
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=labels,
#         color=colors[:num_nodes]  # Apply distinct color to each node
#     ),
#     link=dict(
#         source=sources,
#         target=targets,
#         value=values
#     )
# )])


# fig.update_layout(
#     title_text="🔧 Concern → Job Mapping (Top 3 Repairs per Concern)",
#     font=dict(size=12),
#     height=600,
#     width=1000
# )

# # === Upload to Chart Studio ===
# fig.write_html("concern_job_sankey.html", auto_open=True)

# 3. Part Usage Frequency by Concern

# Clean columns
df["concern"] = df["concern"].astype(str).str.lower().str.strip()
df["job_type"] = df["job_type"].astype(str).str.lower().str.strip()

# Filter irrelevant
irrelevant = ["no technician concerns found.", "no customer concerns found."]
df = df[~df["concern"].isin(irrelevant)]

# 🔟 Top concerns
top_10_concerns = df["concern"].value_counts().nlargest(10).index.tolist()
filtered_df = df[df["concern"].isin(top_10_concerns)]

# 🔁 Top 3 job types per concern
top_jobs = (
    filtered_df.groupby(["concern", "job_type"])
    .size()
    .reset_index(name="count")
    .sort_values(["concern", "count"], ascending=[True, False])
)
top_3_jobs = top_jobs.groupby("concern").head(3).reset_index(drop=True)

# 🔄 Pivot for heatmap format
pivot_df = top_3_jobs.pivot(index="concern", columns="job_type", values="count").fillna(0)

# Title-cased for display
pivot_df.index = pivot_df.index.str.title()
pivot_df.columns = pivot_df.columns.str.title()

# 📊 Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
plt.title("Top 3 Job Types per Concern (Heatmap View)", fontsize=14)
plt.xlabel("Job Type")
plt.ylabel("Concern")
plt.tight_layout()
plt.show()