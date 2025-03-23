import gradio as gr
from function.yolo_seg import *



def _files_predict(model,files,kwargs):
    if kwargs is None:
        return file_predict(model,files)
    else:
        return file_predict(model,files,**kwargs)
def _image_predict(model,image,kwargs):
    if kwargs is None:
        return image_predict(model,image)
    else:
        return image_predict(model,image,**kwargs)

def _video_predict(model,video,kwargs):
    if kwargs is None:
        return video_predict(model,video)
    else:
        return video_predict(model,video,**kwargs)

with gr.Blocks() as demo:
    gr.Markdown("# YOLO seg - Instance segmentation (same as obj_detection")
    with gr.Group():
        model_text = gr.Textbox(
            "In this module,configure your model.(If use pretrain weights,loading will cost time a lot ai first time)",
            label="Model", interactive=False)
        model = gr.State()
        with gr.Row():
            model_version = gr.Dropdown(choices=["yolov8", "yolov9", "yolov10", "yolo11", "yolo12"],
                                        label="model_version", allow_custom_value=True, interactive=True)
            model_size = gr.Dropdown(choices=["n", "t", "s", "m", "l", "x"], label="model_size",
                                     allow_custom_value=True, interactive=True)
        with gr.Row(equal_height=True):
            model_pretrained = gr.Radio(choices=["local", "pretrained"], label="model weights")


            @gr.render(model_pretrained)
            def weights_choice(model_pretrained):
                if model_pretrained == "local":
                    text_tmp = gr.Textbox(model_pretrained, visible=False)
                    model_weights = gr.File(label="model weights", interactive=True)
                    btn_load_model.click(model_load, [model_version, model_size, text_tmp, model_weights], [model],
                                         concurrency_limit=5).then(model_check, model, load_text)
                elif model_pretrained == "pretrained":
                    pretrain_text = gr.Textbox("Pretrained weights used", label="", interactive=False)
                    text_tmp = gr.Textbox(model_pretrained, visible=False)
                    btn_load_model.click(model_load, [model_version, model_size, text_tmp], [pretrain_text, model],
                                         concurrency_limit=5).then(model_check, model, load_text)
                else:
                    pass
        btn_load_model = gr.Button("load model")
        load_text = gr.Textbox(label="model info", value="No model now !", interactive=False)

    with gr.Group():
        detect_text = gr.Textbox("In this module,set the params", label="Params", interactive=False)
        with gr.Row():
            in_type_choice = gr.Dropdown(choices=["image", "video", "files", "webcam", "urls"], label="data_type",
                                         interactive=True)
        other_params = gr.Textbox(label="other params",
                                  placeholder="Input other params to detect(according to JSON format)",
                                  interactive=True)
        params_json = gr.JSON(label="params_json", visible=True)
        other_params.submit(lambda x: x, other_params, params_json, concurrency_limit=5)
    @gr.render(in_type_choice)
    def render_choice(in_type_choice):
        if in_type_choice == "image":
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    image = gr.Image(label="image")
                    btn_predict = gr.Button("Predict")
                with gr.Column(scale=1, min_width=300):
                    result = gr.Image(label="result")
            btn_predict.click(_image_predict, inputs=[model, image, params_json], outputs=result,
                              concurrency_limit=5)
        elif in_type_choice == "video":
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    Video = gr.Video(label="Video")
                    btn_predict = gr.Button("Predict")
                with gr.Column(scale=1, min_width=300):
                    result = gr.Video(label="result", show_download_button=True)
            btn_predict.click(_video_predict, inputs=[model, Video, params_json], outputs=result,
                              concurrency_limit=5)
        elif in_type_choice == "files":
            gr.Markdown(
                "if input is a filefolder or other source,upload your dataset and the results will be return to download")
            with gr.Column(scale=1, min_width=300):
                other_files = gr.Files(label="files", interactive=True)
                return_file = gr.File(label="return file")
                btn_predict = gr.Button("Predict")
            btn_predict.click(_files_predict, inputs=[model, other_files, params_json], outputs=[return_file],
                              concurrency_limit=5)
        elif in_type_choice == "webcam":
            with gr.Row():
                input_img = gr.Image(sources=["webcam"], type="numpy")
                out_img = gr.Image(label="result", streaming=True)
            dep = input_img.stream(_image_predict, [model, input_img, params_json], [out_img],
                                   stream_every=0.1, concurrency_limit=30)
        else:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    hint_text = gr.Textbox(placeholder="pass the right image/video url", interactive=True)
                    btn_predict = gr.Button("Predict")
                other_file = gr.File(label="return file")
            btn_predict.click(_files_predict, [model, hint_text, params_json], [other_file],
                              concurrency_limit=5)

if __name__ == "__main__":
    demo.launch()