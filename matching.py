"""Four image matchers returning corresponding points in input-image pixels."""
from copy import deepcopy

import cv2
import numpy as np


MODEL_NAMES = {
    "sift": "RootSIFT",
    "lightglue": "SuperPoint + LightGlue",
    "loftr": "LoFTR",
    "roma": "RoMa",
}


def empty_points():
    return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)


def valid_points(p0, p1, scores, shape0, shape1, threshold):
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    scores = np.asarray(scores).reshape(-1)
    keep = np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
    keep &= np.isfinite(scores) & (scores >= threshold)
    for points, (h, w) in ((p0, shape0[:2]), (p1, shape1[:2])):
        keep &= (points[:, 0] >= 0) & (points[:, 0] < w)
        keep &= (points[:, 1] >= 0) & (points[:, 1] < h)
    return p0[keep], p1[keep]


class Matcher:
    """Load one model once, then reuse it for every image pair.

    Neural dependencies are optional. Missing packages raise an error; models
    are never silently replaced by a different method.
    """

    def __init__(self, model="sift", device="auto", max_keypoints=2048):
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model}")
        self.kind = model
        self.name = MODEL_NAMES[model]
        self.device = "cpu"
        self.torch = None
        self.settings = {"max_keypoints": max_keypoints}
        if model == "sift":
            self.extractor = cv2.SIFT_create(nfeatures=max_keypoints)
            self.model = cv2.BFMatcher(cv2.NORM_L2)
            self.settings.update(ratio_threshold=0.8, descriptor="RootSIFT")
            return

        import torch
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU runtime or use device='cpu'.")

        if model == "lightglue":
            from lightglue import LightGlue, SuperPoint
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.model = LightGlue(features="superpoint").eval().to(self.device)
            self.settings.update(filter_threshold=0.1, adaptive=True)
        elif model == "loftr":
            from kornia.feature import LoFTR
            from kornia.feature.loftr.loftr import default_cfg, urls
            urls["outdoor"] = urls["outdoor"].replace("http://", "https://", 1)
            config = deepcopy(default_cfg)
            config["match_coarse"]["thr"] = 0.2
            self.model = LoFTR(pretrained="outdoor", config=config).eval().to(self.device)
            self.settings = {"weights": "outdoor", "confidence": 0.2}
        else:
            from romatch import roma_outdoor
            self.model = roma_outdoor(
                device=self.device, use_custom_corr=False, upsample_preds=False
            ).eval()
            self.settings = {"weights": "outdoor", "confidence": 0.2,
                             "max_matches": 2000, "upsample_preds": False,
                             "use_custom_corr": False, "gp_device": "model",
                             "internal_coarse_resolution": 560}

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def __call__(self, image0, image1):
        if self.kind == "sift":
            features = []
            for image in (image0, image1):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = self.extractor.detectAndCompute(gray, None)
                if descriptors is None or len(keypoints) < 2:
                    return empty_points()
                descriptors = np.sqrt(descriptors / (descriptors.sum(1, keepdims=True) + 1e-7))
                features.append((np.float32([k.pt for k in keypoints]), descriptors))
            (p0, d0), (p1, d1) = features
            neighbors = self.model.knnMatch(d0, d1, k=2)
            selected = [a for pair in neighbors if len(pair) == 2
                        for a, b in [pair] if a.distance < 0.8 * b.distance]
            if not selected:
                return empty_points()
            return (p0[[m.queryIdx for m in selected]], p1[[m.trainIdx for m in selected]])

        with self.torch.inference_mode():
            if self.kind == "lightglue":
                features = []
                for image in (image0, image1):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
                    tensor = self.torch.from_numpy(gray)[None].to(self.device)
                    features.append(self.extractor.extract(tensor, resize=None))
                if any(f["keypoints"].shape[1] == 0 for f in features):
                    return empty_points()
                result = self.model({"image0": features[0], "image1": features[1]})
                indices = result["matches"][0]
                p0 = features[0]["keypoints"][0][indices[:, 0]].cpu().numpy()
                p1 = features[1]["keypoints"][0][indices[:, 1]].cpu().numpy()
                return p0, p1

            if self.kind == "loftr":
                tensors, scales = [], []
                for image in (image0, image1):
                    h, w = image.shape[:2]
                    nw, nh = max(32, w // 8 * 8), max(32, h // 8 * 8)
                    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (nw, nh))
                    tensors.append(self.torch.from_numpy(gray.astype(np.float32) / 255)
                                   [None, None].to(self.device))
                    scales.append(np.float32([nw / w, nh / h]))
                result = self.model({"image0": tensors[0], "image1": tensors[1]})
                p0 = result["keypoints0"].cpu().numpy() / scales[0]
                p1 = result["keypoints1"].cpu().numpy() / scales[1]
                return valid_points(p0, p1, result["confidence"].cpu().numpy(),
                                    image0.shape, image1.shape, 0.2)

            from PIL import Image
            images = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in (image0, image1)]
            warp, certainty = self.model.match(*images, device=self.device)
            warp, certainty = warp.reshape(-1, 4), certainty.reshape(-1)
            keep = self.torch.isfinite(warp).all(-1) & self.torch.isfinite(certainty)
            keep &= (certainty >= 0.2) & (warp.abs() <= 1).all(-1)
            warp, certainty = warp[keep], certainty[keep]
            if not len(warp):
                return empty_points()
            if len(warp) > 2000:
                warp, certainty = self.model.sample(warp, certainty, num=2000)
            h0, w0 = image0.shape[:2]
            h1, w1 = image1.shape[:2]
            p0, p1 = self.model.to_pixel_coordinates(warp, h0, w0, h1, w1)
            return valid_points(p0.cpu().numpy(), p1.cpu().numpy(), certainty.cpu().numpy(),
                                image0.shape, image1.shape, 0.2)
