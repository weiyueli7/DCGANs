import zipfile
import os

# Name of the data folder
data_folder = 'data/fake'

# Name of the zip file to be created
zip_file_name = 'fake.zip'

# Create a zip file object
zip_file = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)

# Walk through all the folders and files in the data folder
for root, dirs, files in os.walk(data_folder):
    # Add all the files to the zip file
    for file in files:
        file_path = os.path.join(root, file)
        zip_file.write(file_path, os.path.relpath(file_path, data_folder))

# Close the zip file
zip_file.close()

print(f"The '{data_folder}' folder and its contents have been zipped into '{zip_file_name}'")
