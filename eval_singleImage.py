import numpy as np
import time
import argparse

import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer
import imageio
import skimage

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


use_gpu = True

# list of imagefiles to test on
imagepaths = ['img_5205_33889741601_o-1-e1493074923224.jpg']


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='cocodataset')
    parser.add_argument('--model_path', help='Path to model (.pt) file.', type=str, default='coco_resnet_50_map_0_335_state_dict.pt')
    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # Create the model
    # retinanet = torch.load(parser.model_path)
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet.load_state_dict(torch.load(parser.model_path))
    
    
    if use_gpu:
        device = torch.device('cuda')
        retinanet.cuda()
    else:
        device = torch.device('cpu')
        retinanet.cpu()
    
    retinanet = retinanet.to(device)        


    retinanet.eval()


    transformer=transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    #read and transform image
    for imagepath in imagepaths:
        im = imageio.imread(imagepath)
        #im = skimage.transform.resize(im, (640, 928))
        im = skimage.transform.resize(im, (640, 928))
        img = torch.from_numpy(im).permute(2, 0, 1)
        img = transformer(img).unsqueeze(dim=0)
        
        with torch.no_grad():
            st = time.time()
                        
            scores, classification, transformed_anchors = retinanet(img.float().to(device))
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.5)

    
            img = cv2.cvtColor((255*im).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)
    
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)
    
            cv2.imshow('img', img)
            cv2.waitKey(0)



if __name__ == '__main__':
 main()