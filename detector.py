from mmdet.apis import inference_detector
import mmcv
import pickle

class Detector:
    def __init__(self, model_path, cfg_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.cfg = pickle.load(open(cfg_path, 'rb'))
        self.model.cfg = self.cfg

    def run_inference(self, img, channels='bgr'):
        ''' Pass in img as a numpy array (BGR channel order, as if from cv2) '''
        return inference_detector(self.model, mmcv.imread(img))
