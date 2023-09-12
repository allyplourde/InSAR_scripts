import os
import glob
import zipfile
import shutil
import xml.etree.ElementTree as ET

dir = '/local-scratch/users/aplourde/RS2_ITH/full_scene/raw'
tmpdir = '/local-scratch/users/aplourde/RS2_ITH/full_scene/raw/tmp'
betadir = '/local-scratch/users/aplourde/RS2_ITH/full_scene/raw/constant_beta'

def unzip_and_check_lut(file):

    filename = file.split('/')[-1]

    with zipfile.ZipFile(file, "r") as zf:
        zf.extractall(tmpdir)

    zipdir = os.path.join(tmpdir, filename[:-4])
    target = os.path.join(zipdir,"product.xml")

    tree = ET.parse(target)
    root = tree.getroot()
    schema = root.tag
    schema = schema.split("}")[0]+"}"

    for child in root.findall('.//'+schema+'lutApplied'):
        if child.text == "Constant-beta":
            move_file(filename)

    shutil.rmtree(zipdir)

def move_file(file):

    source_dir = os.path.join(dir, file)
    destination_dir = os.path.join(betadir, file)

    shutil.move(source_dir, destination_dir)


if __name__=="__main__":

    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)

    for file in glob.glob(os.path.join(dir, "*.zip")):
        unzip_and_check_lut(file)

    shutil.rmtree(tmpdir)


