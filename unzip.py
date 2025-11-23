import os
import zipfile

# The folder containing your .zip files (matches your screenshot)
zip_dir = r"/home/harshan/Documents/edl/Face_expression_detection/unzipped/dataset"
# Where you want to extract them (can be the same folder or a new 'extracted' folder)
extract_to = r"/home/harshan/Documents/edl/Face_expression_detection/unzipped/dataset_extracted"

if not os.path.exists(extract_to):
    os.makedirs(extract_to)

# Loop through all files in the directory
for item in os.listdir(zip_dir):
    if item.endswith(".zip"):
        file_path = os.path.join(zip_dir, item)
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract to a folder named after the zip file to keep things organized
                folder_name = os.path.splitext(item)[0]
                target_path = os.path.join(extract_to, folder_name)
                
                zip_ref.extractall(target_path)
                print(f"Extracted: {item}")
        except zipfile.BadZipFile:
            print(f"Error: {item} is a bad zip file.")

print("\nExtraction complete!")