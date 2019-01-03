
import zipfile

my_zip =  zipfile.ZipFile('../dataset/train.zip') # Specify your zip file's name here
storage_path = '../dataset/train'
for file in my_zip.namelist():
    if my_zip.getinfo(file).filename.endswith('_green.png'):
        my_zip.extract(file, storage_path) # extract the file to current folder if it is a text file
