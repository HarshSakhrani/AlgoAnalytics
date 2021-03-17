import zipfile

with zipfile.ZipFile("/root/wav2.zip", 'r') as zip_ref:
  zip_ref.extractall("/root/Unzipped")



#/root/Unzipped/wav/.....
