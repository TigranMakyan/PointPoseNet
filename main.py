import argparse
import torch
import os
import time
from model import PointPoseNet
from data import CustomDataset, build_dataloader
from evaluation import Evaluater
from model import PointPoseNet
from postprocess import PostProcesser
from inference import SingleInference, DataloaderInference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds_dir', '--dataset_directory', type=str, default='/home/tigran/Downloads/tasks')
    parser.add_argument('-dir', '--imagery_dir_name', type=str, default='squirrels_head', 
                        help="the folder directory for mode's inference. If dir_name is all inference will\
                            be done on all the datasets in dataset directory")
    parser.add_argument('-bs', '--batch_size', default=8, type=int)
    parser.add_argument('-im', '--image_path', type=str, help='Image path for inference on the single image.\
                        If image path is not None, inference will be done on that single image',
                        default=None)
    parser.add_argument('-t', '--single_task', default='squirrels_head', type=str, 
                        help='Task for single image inference')
    parser.add_argument('-v', '--visualise', action='store_true', help='In case of you want to see result of \
                        models inference on a single image')
    parser.add_argument('-device', type=str, default='cpu')

    args = vars(parser.parse_args())
    device = torch.device(args['device'])
    main_dir = args['dataset_directory']
    folder_name = args['imagery_dir_name']
    batch_size = args['batch_size']
    single_image_path = args['image_path']
    single_image_task = args['single_task']
    vis = args['visualise']

    if device == 'cuda' and not torch.cuda.is_available():
        print('Cuda is not available. Your device is changed to CPU')
    
    # Creating PostProcessor for predictions
    postprocesser = PostProcesser()
    # Creating model for inference
    model = PointPoseNet(postprocesser=postprocesser, device=device)
    # Creting Evaluater to measure metrics 
    evaluater = Evaluater()

    if single_image_path:
        inference = SingleInference(model=model, evaluater=evaluater, 
                              data=(single_image_path,single_image_task), vis=vis)
        inference.inference()
    else:
        folder_names = os.listdir(main_dir)
        if folder_name in folder_names:
            data_path = os.path.join(main_dir, folder_name)
            # Creating Dataset for inference 
            imagery = CustomDataset(data_path=data_path, device=device)
            #Creating DataLoader for testing
            imagery_loader = build_dataloader(imagery, batch_size=batch_size)
            #Initializing Inference object
            inference = DataloaderInference(model=model, 
                                            evaluater=evaluater, 
                                            data=imagery_loader)
            inference.inference()

        elif folder_name == 'all':
            start_time = time.time()
            accuracy = []
            distance = []
            for folder_name in folder_names:
                data_path = os.path.join(main_dir, folder_name)
                imagery = CustomDataset(data_path=data_path, device=device)
                imagery_loader = build_dataloader(imagery, batch_size=batch_size)
                inference = DataloaderInference(model=model, 
                                            evaluater=evaluater,  
                                            data=imagery_loader)
                acc, dist = inference.inference()
                accuracy.append(acc)
                distance.append(dist)
            exec_time = time.time() - start_time
            print(f'Average accuracy is {(sum(accuracy) / len(accuracy))}')
            print(f'Average distance error is {(sum(distance) / len(distance))}')
            print(f'Execution time for all datasets is {exec_time} seconds')
        else:
            raise NameError('Incorrect directory name')


if __name__ == '__main__':
    main()
