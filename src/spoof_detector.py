import torch
from models.cdcnpp_antispoofing.CDCNs import CDCNpp
from models.DeepPixBisS_anti_spoofing.Model import DeePixBiS
from torchvision import transforms
import cv2
import numpy as np
import os


class SpoofDetector:
    """This module provides a unified interface for different spoof detection methods.
    It supports CDCNpp and DeepPixBiS methods, allowing users to easily switch between them.
    If you want to add a new spoof detection method, create a new class that implements the `predict` method
    and update the `SpoofDetector` class to include it.
    Attributes:
        method (str): The spoof detection method to use ('cdcnpp' or 'deeppixbis').
        detector: An instance of the selected spoof detection class.
    Methods:
        predict(face_img): Predicts the spoof score for the given face image.
    Usage:
        spoof_detector = SpoofDetector(method="deeppixbis")
        score = spoof_detector.predict(face_img)
    """
    def __init__(self, method="cdcnpp", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method

        if self.method == "cdcnpp":
            self.detector = CDCNppDetector(self.device)
        elif self.method == "deeppixbis":
            self.detector = DeepPixBiSDetector(self.device)
        else:
            raise ValueError(f"Unsupported spoof detection method: {self.method}")

    def predict(self, face_img):
        return self.detector.predict(face_img)


class CDCNppDetector:
    def __init__(self, device):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to src/

        model_path = os.path.join(BASE_DIR, "..", "models", "cdcnpp_antispoofing", "CDCNpp_nuaap_100_e-4.pth")
        model_path = os.path.abspath(model_path)
        self.device = device
        self.model = CDCNpp().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path.strip(), map_location=self.device)
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Define preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor()
        ])

    def prepare_input(self, face_img):
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.merge([face_gray])  # HxWx1 to HxWx3 for RGB conversion
        tensor = self.preprocess(face_gray).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, face_img: 'np.ndarray') -> float:
        tensor = self.prepare_input(face_img)
        with torch.no_grad():
            depth_map, *_ = self.model(tensor)
            return torch.mean(depth_map).item()


class DeepPixBiSDetector:
    def __init__(self, device):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to src/

        model_path = os.path.join(BASE_DIR, "..", "models", "DeepPixBisS_anti_spoofing", "DeePixBiS.pth")
        model_path = os.path.abspath(model_path)
        self.device = device
        self.model = DeePixBiS().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, face_img):
        input_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask, binary = self.model(input_tensor)
            prediction = binary.item()
        return prediction


