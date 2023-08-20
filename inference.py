import torch
from tqdm.auto import tqdm
from torchvision import transforms as T
import functools
import time
import cv2

def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time} seconds to run.")
        return result
    return wrapper

class Inference:
    '''
    This is the parent class for inference.
    All other inference classes must be inherited from this class and must 
    realize inference method
    '''
    def __init__(self, model, evaluater, data) -> None:
        self.model = model
        self.evaluater = evaluater
        self.data = data

    def inference(self):
        raise NotImplementedError()


class SingleInference(Inference):
    '''
    This class must be used in case of inference on a single image.
    All preprocessing operations, that you need are realized in this class
    '''
    def __init__(self, model, evaluater, data, vis=True) -> None:
        super().__init__(model, evaluater, data)
        self.vis = vis

    def image_reader(self, image_path):
        '''
        This method return image from the path as torch.tensor and numpy.ndarray
        '''
        try: 
            image = cv2.imread(image_path)
            transform = T.Compose([
                T.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0)
            return image_tensor, image
        except Exception as e:
            print(f'ERROR: Could not read image {image_path}')
            print(e)
            return None

    @timeit_decorator
    def inference(self):
        '''
        Main function for model inference on single image example
        '''
        image_path, task = self.data
        image_tensor, image = self.image_reader(image_path)
        height = image.shape[1]

        with torch.no_grad():
            start_time = time.time()
            prediction = self.model(image_tensor, task)
            elapsed_time = time.time() - start_time
            print(f"Single image inference took {elapsed_time} seconds to run.")
            keypoints = prediction[0]
        # Visualize the keypoints on the image
        if self.vis:   
            print(keypoints)
            x, y = keypoints
            cv2.circle(image, (int(x * height), int(y * height)), 5, (0, 255, 0), -1)                
            cv2.imshow("Keypoint R-CNN", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

class DataloaderInference(Inference):
    '''
    Class must be used for model's inference on datasets for testing and metrics' measurement
    '''
    def __init__(self, model, evaluater, data) -> None:
        super().__init__(model, evaluater, data)
    
    @timeit_decorator
    def inference(self):
        '''
        Main function for model inference on dataloader, that must be created before.
        timeit_decorator is used to measure execution time of this method on the dataset,
        that will be given
        '''
        task = self.data.dataset._name
        print(task)
        correct_count = 0
        total_distances = 0
        with torch.no_grad():
            for i, items in tqdm(enumerate(self.data), total=len(self.data)):
                print(f'Iteration number: ', i+1)
                images, coordinates = items
                model_outputs = self.model(images, task)
                batch_correct_preds, batch_sum_distances = self.evaluater.calculate_accuracy(model_outputs, coordinates)
                correct_count += batch_correct_preds
                total_distances += batch_sum_distances
                print(f'Cumulative accuracy is {(correct_count / ((i+1) * len(items)))}')
                print(f'Cumulative mean distance is {(total_distances / ((i+1) * len(items)))}')
                print(f'Batch accuracy is {(batch_correct_preds / 8)}')
        accuracy = correct_count / len(self.data.dataset)
        mean_error = total_distances / len(self.data.dataset)
        print('Accuracy for this dataset is: ', accuracy)
        print('Mean error for this dataset is: ', mean_error)
        return accuracy, mean_error

    