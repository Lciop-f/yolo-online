import gradio as gr
from matplotlib import interactive
import os
from function.yolo_detecion import *

def token_init(token):
    os.environ['modelscope_token'] = token
def _files_predict(model,files,conf,kwargs):
    if kwargs is None:
        return file_predict(model,files, conf)
    else:
        return file_predict(model,files, conf, **kwargs)
def _image_predict(model,image,conf,kwargs):
    if kwargs is None:
        return image_predict(model,image, conf).plot()
    else:
        return image_predict(model,image,conf,**kwargs).plot()

def _video_predict(model,video,conf,kwargs):
    if kwargs is None:
        return video_predict(model,video, conf)
    else:
        return video_predict(model,video, conf, **kwargs)

def _video_track(model,video,conf,cls,kwargs):
    if kwargs is None:
        return video_track(model,video, conf,cls)
    else:
        return video_track(model,video, conf, cls,**kwargs)

def _image2text(model,image,conf,cls,personal_prompt,kwargs):
    if kwargs is None:
        return image2text(model, image, conf,cls,personal_prompt)
    else:
        return image2text(model, image, conf, cls, personal_prompt,**kwargs)

with gr.Blocks() as demo:

    gr.Markdown(
"""<div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #ddd;">
    <h2>欢迎使用我们的YOLO在线工具！</h2>
    <p>此应用只提供YOLO模型的在线测试功能，可以通过在线检测快速测试您的model，同时也支持利用预训练模型在线检测。</p>
    <p>如需利用您的目标检测模型做更多工作，请打开左侧的功能扩展。</p>
    <p>检测功能依赖Ultralytics，更多参数信息请访问ultralytics的<a href="https://docs.ultralytics.com/zh" target="_blank">官方网站</a>获取更多信息。</p>
</div>"""
    )

    md = gr.Markdown(
"""# YOLO online tools 🚀

use UI to optimize your object detection by YOLO! 🎯

With our intuitive interface, you can easily fine-tune your detection models. 🛠️✨ Whether you're a beginner or an expert, our tools make object detection faster and more accurate. 🚀🔍

Try it out today and see the difference! 💡🌟""")

    with gr.Sidebar():
        gr.Markdown(
            """# YOLO TO：postprocess of your detect result in order to fit more missions"""
        )
        with gr.Group():
            gr.Markdown(
                """<div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3>YOLO Tracker based on ByteTracker</h3>
                    <p>You can utilize the aforementioned results for downstream tasks such as <strong>traffic volume statistics</strong>. The process is remarkably straightforward &mdash; you merely need to input the <strong>original video</strong> to commence.</p>
                </div>"""
            )

            track_block = gr.Checkbox(value=False,label="apply track function")
        with gr.Group():
            gr.Markdown(
                """<div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3>Feature Extraction Function (only image)</h3>
                    <p>This feature allows for further extraction of characteristics from detected objects and returns the feature values in JSON format. If you need to delve deeper into object feature information, enable this function.</p>
                </div>"""
            )

            i2t_block = gr.Checkbox(value=False,label="apply img2txt function")


    with gr.Group():
        model_text = gr.Textbox("In this module,configure your model.(If use pretrain weights,loading will cost time a lot ai first time)",label="Model",interactive=False)
        model = gr.State()
        with gr.Row():
            model_version = gr.Dropdown(choices=["yolov5", "yolov6", "yolov8", "yolov9","yolov10","yolo11","yolo12"], label="model_version",allow_custom_value=True ,interactive=True)
            model_size = gr.Dropdown(choices=["n","t", "s", "m", "l", "x"],label="model_size",allow_custom_value=True ,interactive=True)
        with gr.Row(equal_height=True):
            model_pretrained = gr.Radio(choices=["local", "pretrained"], label="model weights")
            @gr.render(model_pretrained)
            def weights_choice(model_pretrained):
                if model_pretrained == "local":
                    text_tmp = gr.Textbox(model_pretrained,visible=False)
                    model_weights = gr.File(label="model weights",interactive=True)
                    btn_load_model.click(model_load,[model_version,model_size,text_tmp,model_weights],[model],concurrency_limit=None).then(model_check,model,load_text)
                elif model_pretrained == "pretrained":
                    pretrain_text = gr.Textbox("Pretrained weights used",label="",interactive=False)
                    text_tmp = gr.Textbox(model_pretrained, visible=False)
                    btn_load_model.click(model_load,[model_version,model_size,text_tmp],[pretrain_text,model],concurrency_limit=None).then(model_check,model,load_text)
                else:
                    pass
        btn_load_model = gr.Button("load model")
        load_text = gr.Textbox(label="model info",value="No model now !",interactive=False)

    with gr.Group():
        detect_text = gr.Textbox("In this module,set the params",label="Params",interactive=False)
        with gr.Row():
            in_type_choice = gr.Dropdown(choices=["image", "video", "files","webcam","urls"], label="data_type", interactive=True)
            thresh = gr.Slider(0, 1, value=0.5, label="thresh", interactive=True)
        other_params = gr.Textbox(label="other params",placeholder="Input other params to detect(according to JSON format)",interactive=True)
        params_json = gr.JSON(label="params_json",visible=True)
        other_params.submit(lambda x:x, other_params, params_json,concurrency_limit=None)
    @gr.render([in_type_choice,track_block,i2t_block])
    def render_choice(in_type_choice,track_block,i2t_block):
        if in_type_choice == "image":
            if not i2t_block:
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        image = gr.Image(label="image")
                        btn_predict = gr.Button("Predict")
                    with gr.Column(scale=1, min_width=300):
                        result = gr.Image(label="result")
                btn_predict.click(_image_predict, inputs=[model,image,thresh,params_json], outputs=result,concurrency_limit=None)
            else:
                with gr.Column():
                    gr.Markdown("""
                    ## According to your demand,change the prompt.
                    change your prompt carefully in case that your result will have the wrong format
                    """
                                )
                    prompt = gr.Textbox(label="prompt", value="""
                    描述图片人物的相貌，请按照如下格式返回JSON格式字符串：
                    {"年龄":"少年" || "老年","眼镜":"带眼镜" || "不带眼镜"}
                    不要返回多余内容。""", interactive=True)
                    gr.Markdown("""
                    ## Please input your openai api key.
                    The api key will not be explict and be shared with other users
                    """
                                )
                    token = gr.Textbox(label="token", interactive=True)
                    obj = gr.Number(label="obj_id", value=0, interactive=True)
                    token.submit(token_init,token,None)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=300):
                            image = gr.Image(label="image")
                            btn_predict = gr.Button("Predict")
                        with gr.Column(scale=1, min_width=300):
                            result = gr.Image(label="result")
                    ans_json = gr.JSON(label="obj_attr", visible=True)
                    btn_predict.click(_image2text, inputs=[model, image, thresh,obj,prompt,params_json], outputs=[ans_json,result],
                                      concurrency_limit=None)


        elif in_type_choice == "video":
            if track_block:
                gr.Markdown("## If use track func,choose the tracking obj below.Cls is a number according to coco-80 which is car by default.")
                obj = gr.Number(label="obj_id",value=2,interactive=True)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        Video = gr.Video(label="Video")
                        with gr.Row():
                            btn_predict = gr.Button("Predict")
                            btn_track = gr.Button("Track")
                    with gr.Column(scale=1, min_width=300):
                        result = gr.Video(label="result",show_download_button=True)
                btn_predict.click(_video_predict, inputs=[model,Video,thresh,params_json], outputs=result,concurrency_limit=None)
                btn_track.click(_video_track, inputs=[model, Video, thresh,obj, params_json], outputs=result,
                                  concurrency_limit=None)
            else:
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        Video = gr.Video(label="Video")
                        btn_predict = gr.Button("Predict")
                    with gr.Column(scale=1, min_width=300):
                        result = gr.Video(label="result",show_download_button=True)
                btn_predict.click(_video_predict, inputs=[model,Video,thresh,params_json], outputs=result,concurrency_limit=None)

        elif in_type_choice == "files":
            gr.Markdown("if input is a filefolder or other source,upload your dataset and the results will be return to download")
            with gr.Column(scale=1, min_width=300):
                other_files = gr.Files(label="files",interactive=True)
                return_file = gr.File(label="return file")
                btn_predict = gr.Button("Predict")
            btn_predict.click(_files_predict,inputs=[model,other_files,thresh,params_json],outputs=[return_file],concurrency_limit=None)
        elif in_type_choice == "webcam":
            with gr.Row():
                input_img = gr.Image(sources=["webcam"], type="numpy")
                out_img = gr.Image(label="result",streaming=True)
            dep = input_img.stream(_image_predict, [model,input_img,thresh,params_json], [out_img],
                                       stream_every=0.1, concurrency_limit=None)
        else:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    hint_text = gr.Textbox(placeholder="pass the right image/video url",interactive=True)
                    btn_predict = gr.Button("Predict")
                other_file = gr.File(label="return file")
            btn_predict.click(_files_predict, [model,hint_text,thresh,params_json], [other_file],concurrency_limit=None)





if __name__ == "__main__":
    demo.launch()