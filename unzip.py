import zipfile

with zipfile.ZipFile("wav.zip", 'r') as zip_ref:
  zip_ref.extractall("/home/Unzipped")


