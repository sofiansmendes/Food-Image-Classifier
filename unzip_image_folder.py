import zipfile
import os

zip_path = 'data.zip'
extract_path = 'data'
if os.path.isdir(extract_path):
    print('Data Folder Already exists.')
else:
    # open the zip file in read mode
    with zipfile.ZipFile(zip_path, 'r') as zipObj:
        # extract all files into specified directory
        zipObj.extractall(extract_path)