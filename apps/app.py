import gradio as gr
from apps import detect_app,seg_app
import subprocess
import sys

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    detect_app.demo.render()
with demo.route("Segmentation"):
    seg_app.demo.render()

if __name__ == "__main__":
    demo.launch()
    subprocess.run([sys.executable, "release.py"])