import numpy as np
import torch

class PostProcesser:
    '''
    This class will be called after forwarding input data through our model and getting results.
    It is very simple class for all cases and tasks: As our model returns a little complex output,
    so we need some postprocess operations on top of models output. We will take keypoints, and
    ther from those keypoints we will select those ones, that we interested in
    '''
    def process_model_output(self, prediction: list):
        '''
        Since we will use only keypoints from our model, this function will return only
        keypoints from the output of our model without third dimension info
        '''
        result = []
        for dicts in prediction:
            keypoints = dicts['keypoints']
            try:
                other_keys = keypoints[0]
            except:
                print('No keypoints')
                other_keys = torch.ones(17, 3) / 2
            other_keys = other_keys[:, :2]
            result.append(other_keys)
        return result

    def process_head(self, keypoints):
        '''
        Choose keypoints for head
        '''
        out = [keypoint[1:3, :] for keypoint in keypoints]
        return [torch.mean(i, axis=0) for i in out]

    def process_nose(self, keypoints: list):
        '''
        Choose keypoints for nose
        '''
        return [keypoint[0] for keypoint in keypoints]
    
    def process_all(self, keypoints):
        '''
        Choose all keypoints for gemstones
        '''
        return [torch.mean(i, axis=0) for i in keypoints]
    
    def process_tail(self, keypoints):
        '''
        Choose keypoints for tail
        '''
        out = [keypoint[15:, :] for keypoint in keypoints]
        return [torch.mean(i, axis=0) for i in out]
    
    def process_koala_nose(self, keypoints):
        '''
        Choose keypoints for koala's nose
        '''
        out = [keypoint[1:3, :] for keypoint in keypoints]
        return [torch.mean(i, axis=0) for i in out]

    def process_keypoints(self, keypoints, task):
        '''
        main function that will be called for all cases
        '''
        if task == 'squirrels_head':
            return self.process_head(keypoints)
        elif task == 'squirrels_tail':
            return self.process_tail(keypoints)
        elif task == 'the_center_of_the_gemstone':
            return self.process_all(keypoints)
        elif task == 'the_center_of_the_koalas_nose':
            return self.process_koala_nose(keypoints)
        elif task == 'the_center_of_the_owls_head':
            return self.process_head(keypoints)
        elif task == 'the_center_of_the_seahorses_head':
            return self.process_head(keypoints)
        elif task == 'the_center_of_the_teddy_bear_nose':
            return self.process_nose(keypoints)
        elif task == 'all':
            pass
        else:
            raise NameError("This task doesn't exist. Please set correct task")
        
    def process(self, prediction: list, task: str):
        keypoints = self.process_model_output(prediction)
        result_keypoints = self.process_keypoints(keypoints, task)
        return result_keypoints
