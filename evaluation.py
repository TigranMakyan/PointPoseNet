import torch

class Evaluater:
    '''
    Class for model's evaluation
    '''
    def __init__(self, threshold=0.1) -> None:
        '''
        threshold for checking if the output of model is correct or not.
        By default treshold is 0.1, as mentioned in technical requirements
        '''
        self.treshold = threshold

    def calculate_accuracy(self, points: list, gt_points: list):
        '''
        Method for measuring mean accuracy and mean error of the batch of inputs
        '''
        distances = [self.euclidean_distance(point1, point2) for point1, point2 in zip(points, gt_points)]
        print(distances)
        corrects = []
        for i in distances:
            if i < 0.1:
                corrects.append(1)
            else:
                corrects.append(0)
        return sum(corrects), sum(distances)

        
    def calculate_average_distance(self, points, gt_points):
        distances = self.euclidean_distance(points, gt_points)
        return distances / len(points)

    def euclidean_distance(self, points1: torch.Tensor, points2: torch.Tensor):

        if points1.shape != points2.shape:
            print(f'points shape : {points1.shape} {points2.shape}')
            raise ValueError("Points must have the same dimensions")

        squared_diff = torch.pow(points2 - points1, 2)
        # Sum the squared differences along all dimensions and take the square root
        distance = torch.sqrt(torch.sum(squared_diff))
        return distance
    
    