from numpy.core.defchararray import endswith
import zipfile
from ultralytics import YOLO
import cv2
import uuid
import os


def model_load(model_version, model_size, mode,*args):
    model_name = model_version + model_size + "-seg"
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

def image_predict(model,image,**kwargs):
    if model is not None and image is not None:
        result = model.predict(source=image,**kwargs)
        return result[0].plot()


def video_predict(model,video,**kwargs):
    if model is not None and video is not None:
        tmp_path = f"output_{uuid.uuid4()}.mp4"
        result = model.predict(source=video,stream=True,**kwargs)
        W,H,N = next(result).plot().shape
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        output = cv2.VideoWriter(tmp_path, fourcc, 30, (H,W))
        for re in result:
            image = re.plot()
            output.write(image)
        output.release()
        return tmp_path


def file_predict(model,file,**kwargs):
    if model is not None and file is not None:
        results = model.predict(source=file,**kwargs)
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


if __name__ == "__main__":
    model = model_load("yolo11", "n", "pretrained")
