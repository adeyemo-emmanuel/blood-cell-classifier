from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import spaces

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


MODEL_PATH = Path(__file__).parent / "cv01_model_full.pt"
ckpt = torch.load(MODEL_PATH, map_location="cpu")
CLASSES = ckpt["classes"]
IMG_SIZE = ckpt["img_size"]
model = SimpleCNN(len(CLASSES))
model.load_state_dict(ckpt["model_state"])
model.eval()

@spaces.GPU
def classify(img):
    if img is None:
        return None
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    x = ((x - 0.5) / 0.5).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return {c: float(p) for c, p in zip(CLASSES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Single blood cell image"),
    outputs=gr.Label(num_top_classes=8, label="Prediction"),
    title="Blood Cell Classifier",
    description=(
        "A CNN trained on the PBC dataset (8 normal peripheral blood cell types). "
        "Works on single, centred cell images like the dataset; whole-field images or "
        "other stains give unreliable results. Research prototype, not a diagnostic tool."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()