from torch.utils.data import Dataset, DataLoader
import os
import os.path
from PIL import Image
import csv

class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, transform):      

        self.transform = transform
                        
        data_dir = os.path.expanduser(data_dir)
        classes = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}        
        images=[]
        labels=[]
        
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(data_dir, target, 'images')
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append(path)
                    labels.append(class_to_idx[target])

        self.images = images
        self.labels = labels
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.images)
      
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]) 
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]       
        return image, label
    
    def get_labels(self):
        return self.labels

    def get_class_to_idx(self):
        return self.class_to_idx

class TinyImageNetTestDataset(Dataset):

    def __init__(self, data_dir, annot_filename, class_to_idx, transform):      

        self.transform = transform
                        
        data_dir = os.path.expanduser(data_dir)
        
        images = []
        labels = []
        with open(os.path.join(data_dir, annot_filename),'r') as f:
            reader=csv.reader(f,delimiter='\t')
            for imagename, classname, _, _, _, _ in reader:
                images.append(os.path.join(data_dir, 'images', imagename))
                labels.append(class_to_idx[classname])
                
        self.images = images
        self.labels = labels
          
    def __len__(self):
        return len(self.images)
      
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    def get_labels(self):
        return self.labels
