import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64

MATERIALS = ["none", "plastic", "metal", "rock", "wood"]
OBJECT_TYPES = ["none", "ship", "aircraft", "net", "bike", "barrel", "pipeline", "anchor", "tire"]


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MaterialCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))


def load_model(model, filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


autoencoder = load_model(Autoencoder().to(device), "autoencoder (1).pth")
cnn = load_model(MaterialCNN(len(MATERIALS)).to(device), "cnn_classifier.pth")
object_cnn = load_model(MaterialCNN(len(OBJECT_TYPES)).to(device), "object_cnn.pth")

THRESHOLD = 0.01356287021189928


def analyze_sonar_image(image_2d):
    img_tensor = (
        torch.tensor(image_2d, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        reconstructed = autoencoder(img_tensor)
        recon_error = torch.mean((reconstructed - img_tensor) ** 2).item()
        is_anomaly = recon_error > THRESHOLD

        mat_probs = torch.softmax(cnn(img_tensor), dim=1)
        mat_conf, mat_idx = torch.max(mat_probs, dim=1)

        obj_probs = torch.softmax(object_cnn(img_tensor), dim=1)
        obj_conf, obj_idx = torch.max(obj_probs, dim=1)

    return {
        "reconstruction_error": round(recon_error, 5),
        "is_anomaly": bool(is_anomaly),
        "predicted_material": MATERIALS[mat_idx.item()] if is_anomaly else "none (normal seafloor)",
        "material_confidence": round(mat_conf.item(), 3),
        "predicted_object": OBJECT_TYPES[obj_idx.item()] if is_anomaly else "none",
        "object_confidence": round(obj_conf.item(), 3),
    }


def gradio_predict(input_image):
    if input_image is None:
        return "Please upload a sonar image.", None, None

    if len(input_image.shape) == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = input_image

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    result = analyze_sonar_image(normalized)

    if result["is_anomaly"]:
        status_md = (
            "### RED ANOMALY DETECTED\n"
            f"**Object Type:** {result['predicted_object'].upper()} "
            f"({result['object_confidence'] * 100:.1f}%)\n\n"
            f"**Material:** {result['predicted_material'].upper()} "
            f"({result['material_confidence'] * 100:.1f}%)"
        )
    else:
        status_md = "### GREEN NORMAL SEAFLOOR\nNo debris detected"

    details_md = (
        f"**Reconstruction Error:** {result['reconstruction_error']}\n\n"
        f"**Threshold:** {THRESHOLD:.5f}"
    )

    display_img = resized.astype(np.uint8)
    return status_md, details_md, display_img


with gr.Blocks(title="Marine Debris Detection") as demo:
    banner_path = str(BASE_DIR / "banner.png")
    if os.path.exists(banner_path):
        gr.Image(value=banner_path, show_label=False, container=False, height=200)

    gr.Markdown(
        "Upload a side-scan sonar image to detect debris, material type, and object type."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Sonar Image", type="numpy")
            submit_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column():
            output_status = gr.Markdown()
            output_details = gr.Markdown()
            output_img = gr.Image(label="Processed Image (64x64)")

    submit_btn.click(
        fn=gradio_predict,
        inputs=input_img,
        outputs=[output_status, output_details, output_img],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
