import gdown
url = "https://drive.google.com/uc?id=1OBN2bBFJCFw6MzKOk3mFT-r_oTqMaY9Q"
output = 'wav.zip'
gdown.download(url, output, quiet=False)
