from mmdet.apis import init_detector, inference_detector

class Predicter():
    def __init__(self, 
        config_path = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 
        checkpoint_path = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', 
        device = 'cuda:0'):
        
        config_file = config_path
        # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
        # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
        checkpoint_file = checkpoint_path
        device = device
        # 初始化检测器
        self.model = init_detector(config_file, checkpoint_file, device=device)

    def inference(self, img):
        result = inference_detector(self.model, img)
        result = self.model.show_result(img,result)
        return result