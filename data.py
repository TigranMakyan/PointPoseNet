from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# import jpeg4py
import json
import os
import torch
import cv2

class CustomDataset(Dataset):
    '''
    Custom class for data processing in format of torch.utils.data.Dataset
    '''
    def __init__(self, data_path: str, device: str) -> None:
        self.device = device
        self.data_path = data_path
        self._name = os.path.basename(data_path)
        self.data = dict()
        self.json_files_all = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path) if file.endswith('.json')]
        self.json_files = [file for file in self.json_files_all if os.stat(file).st_size != 0]
        self.image_files = [file.replace('.json', '.jpg') for file in self.json_files]
        self.data = [{'image': i, 'coordinates': j} for i, j in zip(self.image_files, self.json_files)]
        print(f'len dataset: {len(self.image_files)}, {len(self.json_files)}')


    def __getitem__(self, index):
        img_path = self.data[index]['image']
        coords_path = self.data[index]['coordinates']
        image = self.image_loader(img_path)
        image = self.image_preprocess(image)
        coordinates = self.coordinates_reader(coords_path)
        if coordinates is None:
            return None
        return image, coordinates


    def __len__(self):
        return len(self.data)

    def image_loader(self, image_path: str):
        '''
        This method return image from the path as numpy.ndarray
        '''
        try: 
            #return jpeg4py.JPEG(image_path).decode()
            image = cv2.imread(image_path)
            return image
        except Exception as e:
            print(f'ERROR: Could not read image {image_path}')
            print(e)
            return None
        
    def image_preprocess(self, image):
        '''
        The method takes as argument image in format of numpy.ndarray.
        The image will be standartized, normalized(optionally) and converted to torch.tensor on the mentioned device
        Return torch.tensor (on device)
        '''
        transform = T.Compose([
            T.ToTensor(),
            # T.Grayscale()
            T.Lambda(lambda x: x.to(self.device)),
            # T.Normalize(mean = [0.351, 0.424, 0.518], std = [0.272, 0.260, 0.268])
        ])
        return transform(image)
    
    def coordinates_reader(self, json_file_path):
        '''
        This method returns tensor of coordinates of points  
        Example:
            torch.Tensor([0.5321502685546875, 0.3208160400390625])
        '''
        try:
            with open(json_file_path, 'r') as f:
                coords = json.load(f)
                return torch.tensor([coords[0]['x'], coords[0]['y']])
        except:
            print('JSON file is empty')
            return None
        
def build_dataloader(dataset, batch_size=32):
    '''
    function for dataloader building
    '''
    dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dl
        
    

