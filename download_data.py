import gdown

# Replace the file ID in the URL with the actual file ID from your Google Drive link
file_id = '1YncwdJEb6VEfC0GkWVydBiOTyedV0uJ0'
url = f'https://drive.google.com/uc?id={file_id}'

# Specify the destination file path
output = 'data.zip'

# Download the file
gdown.download(url, output, quiet=False)
