import sys 
import cv2
import os
import time 
from datetime import datetime
import numpy as np
import torch
import time
import random
import torch.backends.cudnn as cudnn
import paho.mqtt.client as mqtt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.augmentations import letterbox


publish_topic = 'cuidapp'
message_stack = []

def load_model(model):
        weights="/home/xavier1/Desktop/GALP_Project/models/" + model # The string should be the absolute path to the models folder
        device_number = 0
        half = False
        if model=='dirt_best.pt':
            imgsz = 640
        else:
            imgsz = 256

        
        set_logging()
        device = select_device(device_number)
        half &= device.type != 'cpu'  

        
        model = attempt_load(weights, map_location=device)  
        stride = int(model.stride.max())  
        imgsz = check_img_size(imgsz, s=stride)  
        names = model.module.names if hasattr(model, 'module') else model.names  
        if half:
            model.half()
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
        
        return device, half, model, stride, imgsz, names

def inference(device, half, model, stride, imgsz, names, algorithm):
        images = []
        conf_thres = 0.50
        iou_thres = 0.45
        max_det = 1000
        hide_labels = False
        hide_conf = False
        classes = None
        agnostic_nms = False
        line_thickness = 3
        cudnn.benchmark = True

        if(algorithm=='dirt'):
            images_dir = '/home/xavier1/Desktop/data_images/dirt/test/' # The string should be the absolute path to the dirt images folder presented in data
        elif(algorithm=='damage'):
            images_dir = '/home/xavier1/Desktop/data_images/damage/test/' # The string should be the absolute path to the damage images folder presented in data
        else:
            pass

        image = random.choice(os.listdir(images_dir))
        image_path = images_dir + image
        image_to_infer = cv2.imread(image_path)
        images.append(image_to_infer)


        labels_detected = []
        frame_copy = images[0].copy()
        frame = [letterbox(x, imgsz, auto=True, stride=stride)[0] for x in images]
        frame = np.stack(frame, 0)
        frame = frame[..., ::-1].transpose((0, 3, 1, 2))  
        frame = np.ascontiguousarray(frame)
        frame = torch.from_numpy(frame).to(device)
        frame = frame.half() if half else frame.float()  
        frame /= 255.0  
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)

        
        pred = model(frame,
                        augment=False,
                        visualize=False)[0]

        
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        
        for i, det in enumerate(pred):  
            if len(det):
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], frame_copy.shape).round()

                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box(xyxy, frame_copy, label=label, color=colors(c, True), line_thickness=line_thickness)
                    labels_detected.append(names[c])
        
        
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

        return frame_copy, labels_detected, det



def callback(client, userdata, message):
    message = message.payload.decode("utf-8")
    print(message)

    if(message=='[XDK_DIRT]:0'):

        
        message_stack.append('[JETSON_DIRT]')
        image, labels, det = inference(device_dirt, half_dirt, model_dirt, stride_dirt, imgsz_dirt, names_dirt, 'dirt')
        
        image = cv2.resize(image, (1600,900))
        cv2.imshow('image', image)
        

        for label in labels:
            if label=='Wheels Dirt':
                
                message_stack.append('[JETSON_DIRT_R]')
                
            elif label=='Wheels Clean':
                pass
            elif label=='Lateral Dirt':
                message_stack.append('[JETSON_DIRT_L]')
            elif label=='Lateral Clean':
                pass
            elif label=='Top Dirt':
                message_stack.append('[JETSON_DIRT_T]')
            elif label=='Top Clean':
                pass
            else:
                pass

    elif(message=="[XDK_DMG]:0"):

        message_stack.append('[JETSON_DMG]')
        image, labels, det = inference(device_damage, half_damage, model_damage, stride_damage, imgsz_damage, names_damage, 'damage')
        
        image = cv2.resize(image, (1600,900))
        cv2.imshow('image', image)

        for label in labels:
            if label=='Scratch':
                message_stack.append('[JETSON_DMG_R]')
            elif label=='Broken Glass':
                message_stack.append('[JETSON_DMG_V]')
            elif label=='Deformation':
                message_stack.append('[JETSON_DMG_R]')
            elif label=='Broken':
                message_stack.append('[JETSON_DMG_P]')
            else:
                pass

    else:
        pass


if __name__ == "__main__":

	# Here you set the configurations for the MQTT broker
	# MQTT was used to integrate the smartphone, IoT sensor and Jetson Inference

        broker= "192.168.0.196"
        client= mqtt.Client("cuida-client")
        client.username_pw_set(username="default", password="default")
        client.connect(broker)
        client.loop_start()
        client.subscribe('cuida')
        client.on_message = callback

        device_dirt, half_dirt, model_dirt, stride_dirt, imgsz_dirt, names_dirt = load_model('dirt_best.pt')
        device_damage, half_damage, model_damage, stride_damage, imgsz_damage, names_damage = load_model('damage_best.pt')

        while True:
            if len(message_stack)==0:
                pass
            else:
                for message in message_stack:
                    client.publish(publish_topic, message)
                    time.sleep(3)
                message_stack=[]
            
            if cv2.waitKey(30) & 0xff == ord('q'):
                client.disconnect()
                client.loop_stop()
                cv2.destroyAllWindows()
                break
