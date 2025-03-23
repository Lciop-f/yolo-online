import torch
from numpy.core.defchararray import endswith
import zipfile
from ultralytics import YOLO
import cv2
import uuid
import os
import argparse
import numpy as np
from function.track import Mytracker,make_parser
from function.image2text import img2text



def model_load(model_version, model_size, mode,*args):
    model_name = model_version + model_size
    if mode == "local" and endswith(args[0], suffix=".pt"):
        try:
            model = YOLO(args[0])
            return model
        except FileNotFoundError:
            return "model weights not found"
    elif mode == "pretrained":
        weights_path = model_name + ".pt"
        try:
            model = YOLO(weights_path)
            return "Finishing load pretrain model",model
        except FileNotFoundError:
            return "Pretrained weights not exists, please check your model version to confirm",None
    else:
        pass

def model_check(model):
    if isinstance(model,YOLO):
        return "Model loaded succeed !"
    else:
        return "Model loaded failed !"

def image_predict(model,image,conf,**kwargs):
    if model is not None and image is not None:
        result = model.predict(source=image,conf=conf,**kwargs)
        return result[0]


def video_predict(model,video,conf,**kwargs):
    if model is not None and video is not None:
        tmp_path = f"output_{uuid.uuid4()}.mp4"
        result = model.predict(source=video,conf=conf,stream=True,**kwargs)
        W,H,N = next(result).plot().shape
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        output = cv2.VideoWriter(tmp_path, fourcc, 30, (H,W))
        for re in result:
            image = re.plot()
            output.write(image)
        output.release()
        return tmp_path


def file_predict(model,file,conf,**kwargs):
    if model is not None and file is not None:
        results = model.predict(source=file,conf=conf,**kwargs)
        temp_dir = f"temp_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        for i, re in enumerate(results):
            image = re.plot()
            image_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            cv2.imwrite(image_path, image)
        zip_path = f"output_{uuid.uuid4()}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, temp_dir))
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

        return zip_path

def video_track(model,video,conf,cls=2,**kwargs):
    if model is not None and video is not None:
        tmp_path = f"output_{uuid.uuid4()}.mp4"
        tracker = Mytracker(make_parser().parse_args())
        result = model.predict(source=video,conf=conf,stream=True,classes=[cls],**kwargs)
        W,H,N = next(result).plot().shape
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        output = cv2.VideoWriter(tmp_path, fourcc, 30, (H,W))
        for re in result:
            dets = torch.cat([re.boxes.xyxy, re.boxes.conf.unsqueeze(1)], dim=1)
            image_size = (H, W)
            image = re.plot(conf=False, boxes=False,labels=False,probs=False)
            image_target = tracker.track(dets, image_size, image_size, image)
            output.write(image_target)
        output.release()
        return tmp_path


def image2text(model,image,conf,cls,personal_prompt,**kwargs):
    if model is not None and image is not None:
        result = model.predict(source=image,conf=conf,classes=[cls],**kwargs)
        ans_dict = {}
        for i,re in enumerate(result[0].boxes.xyxy):
            x1,y1,x2,y2 = np.array(re).astype("int")
            img = image[y1:y2, x1:x2,:]
            ans_dict[f"obj{i}"] = img2text(img,personal_prompt)
        return ans_dict,result[0].plot()

if __name__ == "__main__":
    s = model_load("yolo12","n","local","/Users/caixinyuan/PycharmProjects/yolo-online/y11.pt")
    i = cv2.imread("//test1.png")
    ans,image = image2text(s,i,conf=.5,cls=0,personal_prompt="""描述图片人物的相貌，请按照如下格式返回JSON格式字符串：\n{"年龄":"少年" || "老年","眼镜":"带眼镜" || "不带眼镜"}""")
    print(ans)




