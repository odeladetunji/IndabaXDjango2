import cv2
from processing import *
import glob
import os
import random
from tqdm import tqdm

def get_file_name_dir(rootDir, ext):
 
    # Return file names with certain extension

    return glob.glob1(rootDir, "*." + ext)


def resize_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (512,512), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = normalize_staining(img)
    img_aug = hematoxylin_eosin_aug(img_norm)
    cv2.imwrite(os.path.join(output_dir, filename.split('/')[-1]), img_aug)
    # take it back to PIL image

def augment(input_path, input_extension):

    inputs_files = sorted(get_file_name_dir(input_path, input_extension))
    cont = 1

    for file_name in inputs_files:
        print('Preprocessing: ' + file_name + ': ' + str(cont))
        cont += 1
        img = input_path + file_name
        im_in = cv2.imread(img)
        
        image = HorizontalFlip(im_in)
        cv2.imwrite(input_path + 'HF_' +file_name , image)
        
        image = VerticalFlip(im_in)
        cv2.imwrite(input_path + 'VF_' +file_name , image)
        
        image = RandomRotate(im_in)
        cv2.imwrite(input_path + 'RR_' +file_name , image)
        
        image = RandomContrast(im_in)
        cv2.imwrite(input_path + 'RC_' +file_name , image)

        image = RandomBrightness(im_in)
        cv2.imwrite(input_path + 'RB_' +file_name , image)
        
        image = RandomHueSaturationValue(im_in)
        cv2.imwrite(input_path + 'HSV_' +file_name , image)
        print("*"*30)
        print("\nDone")


PATH = "ICIAR2018_BACH_Challenge/Photos/"
SIZE = 512
classes = ["Benign", "InSitu",  "Invasive",  "Normal"]

for clas in classes:
    print("Generating Validation for: ", clas)

    # some files are not images so need to filter them
    file_list = glob.glob(f"{PATH}{clas}/*.tif")
    
    random.seed(230)
    file_list.sort()
    random.shuffle(file_list)
    
    split = int(0.8 * len(file_list))
    train_filenames = file_list[:split]
    test_filenames = file_list[split:]
    
    filenames = {'train': train_filenames,
                'test': test_filenames} 
  
    for split in ['train', 'test']:        
        output_dir = "dataset/" + split + '/' + clas
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir, size=SIZE)


augment('dataset/train/InSitu/', 'tif')
augment('dataset/train/Invasive/', 'tif')
augment('dataset/train/Normal/', 'tif')
augment('dataset/train/Benign/', 'tif')


