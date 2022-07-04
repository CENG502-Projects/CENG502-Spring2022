from configuration import *
import shutil

dataset_labels = {}

def copy_labelled_images(label,outputFolder):
    global dataset_labels
    
    train_path = test_dataset_path + "/ImageSets/Main/{}_test.txt".format(label)
    count = 0
    with open(train_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(" ")
            if len(parts) == 3 and int(parts[-1])==1:
                count += 1
                image_name = parts[0]
                src_file = test_dataset_path + "/JPEGImages/{}.jpg".format(image_name)
                print("{} - {}".format(image_name,label))
                shutil.copy(src_file,outputFolder)
                if image_name in dataset_labels.keys():
                    dataset_labels[image_name][voc_labels[label]] = 1
                else:
                    dataset_labels[image_name] = [0]*len(voc_labels.keys())
                    dataset_labels[image_name][voc_labels[label]] = 1

for label  in voc_labels.keys():
    copy_labelled_images(label, test_dataset_output_path + "/images/") 

for image_name in dataset_labels.keys():
    label_path = test_dataset_output_path + "/labels/{}.txt".format(image_name)
    with open(label_path,"w") as fp:
        for elem in dataset_labels[image_name]:
            fp.write("{}\n".format(elem))
