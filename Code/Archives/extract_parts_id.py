# Required libraries
import fitz  # PyMuPDF
import re
import os
import pandas as pd

# Define folder where invoice PDFs are stored
pdf_folder = r"C:\\Users\\mruga\\Desktop\\New folder\\Capstone Project\\Capstone Invocies\\Raw_Invoices\\Raw_Invoices"

# Limit the number of PDFs for testing (set to None to process all)
TEST_LIMIT = 5

# Regex pattern to capture part numbers with at least one digit and dash
part_number_pattern = re.compile(r"\b(?:[A-Z]*\d+-\d+|\d{2,}-[A-Z]+|[A-Z]+-\d{2,}|\d{5,})\b")

# List of known non-part keywords to filter out
non_part_keywords = {"REORDER", "VEHICLE", "JOBS", "CHRISTOPHER", "COLLAPSE", "ISSUES"}

# Helper function to extract detailed part info from a single PDF
def extract_part_info_from_pdf(pdf_path):
    part_entries = []
    doc = fitz.open(pdf_path)
    for page in doc:
        lines = page.get_text().split('\n')
        for i in range(len(lines)):
            line = lines[i].strip()
            number_matches = part_number_pattern.findall(line)
            if number_matches:
                part_number = number_matches[0].strip()

                # Skip bad tokens
                if part_number.upper() in non_part_keywords or part_number.isalpha():
                    continue

                qty = cost = retail = total = part_name = vendor = ""

                # Try parsing part info from nearby lines
                if i >= 1:
                    part_name = lines[i - 1].strip()
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if 'WORLDPAC' in next_line.upper():
                        vendor = 'WORLDPAC'
                    elif any(char.isdigit() for char in next_line):
                        cost_line = next_line.replace('$', '').replace(',', '').split()
                        if len(cost_line) >= 4:
                            try:
                                qty = cost_line[0]
                                cost = cost_line[1]
                                retail = cost_line[2]
                                total = cost_line[3]
                            except:
                                pass

                # Only include rows with cost or qty to avoid garbage
                if qty or cost or part_name:
                    part_entries.append({
                        "part_number": part_number,
                        "part_name": part_name,
                        "vendor": vendor,
                        "qty": qty,
                        "cost_per_unit": cost,
                        "retail_per_unit": retail,
                        "total_cost": total
                    })
    return part_entries

# Extract part info from PDFs and map to ro_number
part_data = []
failed_files = []
successful_count = 0

print("Looking for PDFs in:", pdf_folder)
all_files = os.listdir(pdf_folder)
pdf_files = [f for f in all_files if f.endswith(".pdf")]

if TEST_LIMIT:
    pdf_files = pdf_files[:TEST_LIMIT]

total_files = len(pdf_files)
current_count = 0

for filename in pdf_files:
    current_count += 1
    print(f"Processing {current_count}/{total_files}: {filename}")
    pdf_path = os.path.join(pdf_folder, filename)
    ro_number = os.path.splitext(filename)[0].replace("repair_order_", "")
    part_entries = extract_part_info_from_pdf(pdf_path)
    if part_entries:
        successful_count += 1
        for entry in part_entries:
            entry.update({"ro_number": ro_number, "source_file": filename})
            part_data.append(entry)
    else:
        failed_files.append(filename)

# Convert to DataFrame and save separately
parts_df = pd.DataFrame(part_data)
parts_output_path = r"C:\\Users\\mruga\\Desktop\\New folder\\Capstone Project\\AutoShopIQ_Repair_chatbot\\Data\\Invoices_converted_csv\\parts_detailed_extracted_TEST.csv"
parts_df.to_csv(parts_output_path, index=False)

# Output summary info
print("Detailed part extraction completed.")
print(f"Total PDFs processed: {total_files}")
print(f"Successfully extracted part data from: {successful_count} PDFs")
print(f"Total PDFs with no part matches found: {len(failed_files)}")
if failed_files:
    print("Files without part matches:")
    for file in failed_files:
        print(f"- {file}")