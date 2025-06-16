# Required libraries
import fitz  # PyMuPDF
import re
import os
import pandas as pd

# Define folder where invoice PDFs are stored
pdf_folder = "./invoices"  # Update this path

# Regex pattern for part numbers (adjust based on your real pattern)
part_number_pattern = re.compile(r"\b(?:[A-Z0-9]{3,}-[A-Z0-9]{2,}|\d{3}-\d{5}|[A-Z]{2,}\d{4,})\b")

# Helper function to extract part numbers from a single PDF
def extract_part_numbers_from_pdf(pdf_path):
    part_numbers = set()
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        matches = part_number_pattern.findall(text)
        part_numbers.update(matches)
    return list(part_numbers)

# Extract part numbers from all PDFs and map to ro_number (from filename or embedded text)
part_data = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        ro_number = os.path.splitext(filename)[0]  # Assuming filename is like 12345.pdf
        parts = extract_part_numbers_from_pdf(pdf_path)
        for part in parts:
            part_data.append({"ro_number": ro_number, "part_number": part})

# Convert to DataFrame
parts_df = pd.DataFrame(part_data)

# Load the structured repair data
structured_df = pd.read_csv("structured_repair_data.csv")

# Merge part numbers into the structured data (many-to-many possible)
updated_df = structured_df.merge(parts_df, on="ro_number", how="left")

# Save the updated CSV
updated_df.to_csv("structured_repair_data_with_part_ids.csv", index=False)

print("Part number extraction and merge completed.")
