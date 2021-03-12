import zipfile

with zipfile.ZipFile("/home/wav.zip", 'r') as zip_ref:
  zip_ref.extractall("/home/Unzipped")


