import zipfile

with zipfile.ZipFile("/root/wav.zip", 'r') as zip_ref:
  zip_ref.extractall("/root/Unzipped")



#/root/Unzipped/wav/.....
