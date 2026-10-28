

import gc
import os
import cv2
import json
import torch
import torch.nn as nn
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from src import open_clip
from few_shot import memory
from dataset import *
import logging
from tqdm import tqdm
from logging import getLogger


# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.open_clip import tokenizer






import warnings

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def vis(pathes, anomaly_map, img_size, save_path, cls_name):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)

class prompt_order():
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list =[
        "a cropped photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of the {}.",
        "a dark photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.",
        "a blurry photo of a {}.",
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.",
        "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection."
        ]
    def prompt(self, class_name):
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template

class patch_scale():
    def __init__(self, image_size):
        self.h, self.w = image_size
 
    def make_mask(self, patch_size = 16, kernel_size = 16, stride_size = 16): 
        self.patch_size = patch_size
        self.patch_num_h = self.h//self.patch_size
        self.patch_num_w = self.w//self.patch_size
        ###################################################### patch_level
        self.kernel_size = kernel_size//patch_size
        self.stride_size = stride_size//patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype = torch.float32).reshape((1,1,self.patch_num_h, self.patch_num_w))
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy



simple_tokenizer = tokenizer.SimpleTokenizer()


class CLIP_AD(nn.Module):
    def __init__(self,model_name = 'ViT-B-16-plus-240'):
        super(CLIP_AD, self).__init__()
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained='laion400m_e31')
        self.mask = patch_scale((240,240))
    def multiscale(self):
        pass
    
    def encode_text(self, text):
        return self.model.encode_text(text)
    def encode_image(self, image, patch_size, mask=True):
        if mask:
            b, _, _, _ = image.shape
            large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().cuda()
            mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().cuda()
            tokens_list, class_tokens, patch_tokens = self.model.encode_image(image, [large_scale,mid_scale], proj = False)
            large_scale_tokens, mid_scale_tokens = tokens_list[0], tokens_list[1]
            return large_scale_tokens, mid_scale_tokens, patch_tokens.unsqueeze(2), class_tokens, large_scale, mid_scale

def compute_score(image_features, text_features):
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)

    return text_probs


def compute_sim(image_features, text_features):

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    simmarity = (torch.bmm(image_features.squeeze(2), text_features)/0.07).softmax(dim=-1)
    return simmarity
def harmonic_aggregation(score_size, simmarity, mask):
    b, h, w = score_size
    # Each column of mask describes a sliding-window token.  Build the
    # patch-to-window incidence matrix once and aggregate all patches at once.
    # The former implementation executed h*w Python loops and torch.isin calls
    # per batch, which dominated 1-shot evaluation time.
    # Retain the original float64 aggregation for reproducible metric values.
    simmarity = simmarity.double()
    patch_ids = torch.arange(1, h * w + 1, device=simmarity.device, dtype=mask.dtype)
    incidence = (patch_ids[:, None, None] == mask.to(simmarity.device)[None]).any(dim=1)
    counts = incidence.sum(dim=1).to(simmarity.dtype)
    harmonic_sum = torch.reciprocal(simmarity) @ incidence.T.to(simmarity.dtype)
    return (counts.unsqueeze(0) / harmonic_sum).reshape(b, h, w)


def safe_image_metrics(labels, scores):
    """Return image-level metrics, or NaN when a split has one label class."""
    if np.unique(labels).size < 2:
        return float("nan"), float("nan"), float("nan")

    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = (2 * precision * recall) / (precision + recall)
    return (
        roc_auc_score(labels, scores),
        np.max(f1[np.isfinite(f1)]),
        average_precision_score(labels, scores),
    )

def harmonic(data):

    scale = data.shape[1]
    Denominator = 0
    for idx in range(scale):
        mask = torch.ones(scale)
        mask[idx] = 0
        mask = (mask == 1)
        Denominator += torch.prod(data[:, mask], dim = -1)
    numerator = torch.prod(data, dim = -1)
    return scale * numerator/Denominator

def harmonic_mean_deconv(scores, stride, kernel_size, padding = 0):
    N, C, H, W = scores.shape
    H_out = (H - 1)*stride + kernel_size - 2*padding
    W_out = (W - 1)*stride + kernel_size - 2*padding
    overlap = (kernel_size - stride) // 2
    result = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    count = 0
                    harmonic_sum = 0
                    for u in range(i*stride-overlap, i*stride+overlap+1):
                        for v in range(j*stride-overlap, j*stride+overlap+1):
                            if u >= 0 and u < H_out and v >= 0 and v < W_out:
                                count += 1
                                weight = 1
                                harmonic_sum += weight / scores[n, c, i, j]
                    result[n, c, i*stride, j*stride] = count / harmonic_sum
    return result

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def few_shot(memory, token, class_name):
    retrive = []
    for i in class_name:
        L, N, D = memory[i].shape
        retrive.append(memory[i].permute(2, 1, 0).reshape(D,-1)) # D NL
    retrive = torch.stack(retrive)# B D NL
     #B D L 
    M = 1/2 * torch.min(1.0 - torch.bmm(F.normalize(token.squeeze(2), dim = -1), F.normalize(retrive, dim = 1)), dim = -1)[0]
    return M

@torch.no_grad()
def prepare_text_future(model, obj_list):
    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    text_generator = prompt_order()

    for i in obj_list:

        normal_description, abnormal_description = text_generator.prompt(i)

        normal_tokens = tokenizer.tokenize(normal_description)
        abnormal_tokens = tokenizer.tokenize(abnormal_description)
        normal_text_features = model.encode_text(normal_tokens.cuda()).float()
        abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True) 
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)
        Mermory_avg_normal_text_features.append(avg_normal_text_features)
        Mermory_avg_abnormal_text_features.append(avg_abnormal_text_features)
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)       
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)  
    return (
        Mermory_avg_normal_text_features.detach(),
        Mermory_avg_abnormal_text_features.detach(),
    )


class SingleClassEvaluator:
    """Reuse model, text features, and test data while evaluating 1-shot supports."""

    def __init__(self, args):
        if args.class_name is None:
            raise ValueError("SingleClassEvaluator requires --class_name.")

        self.args = args
        self.img_size = args.image_size
        self.dataset_dir = args.data_path
        self.dataset_name = args.dataset
        self.class_name = args.class_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.patch_size = 16

        class_names = {
            "mvtec": [
                "carpet", "bottle", "hazelnut", "leather", "cable", "capsule",
                "grid", "pill", "transistor", "metal_nut", "screw", "toothbrush",
                "zipper", "tile", "wood",
            ],
            "visa": [
                "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
            ],
        }
        if self.dataset_name not in class_names or self.class_name not in class_names[self.dataset_name]:
            raise ValueError(f"Class {self.class_name} not found in {self.dataset_name}.")

        self.model = CLIP_AD(args.model).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])
        # Match the repository's VisaDataset evaluation path: binary masks are
        # encoded as 0/255, resized with torchvision's default interpolation,
        # then thresholded at 0.5 below.
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])
        self.preprocess = self.model.preprocess
        self.preprocess.transforms[0] = transforms.Resize(
            size=(self.img_size, self.img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        )
        self.preprocess.transforms[1] = transforms.CenterCrop(self.img_size)

        with open(os.path.join(self.dataset_dir, "meta.json"), encoding="utf-8") as file:
            test_samples = json.load(file)["test"][self.class_name]

        evaluator = self

        class ClassTestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(test_samples)

            def __getitem__(self, index):
                info = test_samples[index]
                image = Image.open(os.path.join(evaluator.dataset_dir, info["img_path"])).convert("RGB")
                image = evaluator.preprocess(image)
                mask_path = info["mask_path"]
                if mask_path:
                    raw_mask = Image.open(os.path.join(evaluator.dataset_dir, mask_path)).convert("L")
                    mask = Image.fromarray(
                        (np.asarray(raw_mask) > 0).astype(np.uint8) * 255,
                        mode="L",
                    )
                    mask = evaluator.mask_transform(mask)
                else:
                    mask = torch.zeros((1, image.shape[1], image.shape[2]))
                return {
                    "img": image,
                    "img_mask": mask,
                    "anomaly": info["anomaly"],
                }

        test_dataloader = torch.utils.data.DataLoader(
            ClassTestDataset(), batch_size=args.batch_size, shuffle=False
        )
        self.normal_text, self.anomaly_text = prepare_text_future(self.model, [self.class_name])
        self.test_batches = self._cache_test_features(test_dataloader)

    @torch.inference_mode()
    def _cache_test_features(self, test_dataloader):
        """Encode deterministic test images once and keep the results on CPU."""
        batches = []
        for items in test_dataloader:
            images = items["img"].to(self.device)
            masks = (items["img_mask"] > 0.5).to(items["img_mask"].dtype)
            batch_size, _, height, width = images.shape
            large_tokens, mid_tokens, patch_tokens, class_tokens, large_scale, mid_scale = \
                self.model.encode_image(images, self.patch_size)
            batches.append({
                "masks": masks.detach().cpu(),
                "anomaly": items["anomaly"].detach().cpu(),
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "large_tokens": large_tokens.detach().cpu(),
                "mid_tokens": mid_tokens.detach().cpu(),
                "patch_tokens": patch_tokens.detach().cpu(),
                "class_tokens": class_tokens.detach().cpu(),
                "large_scale": large_scale.detach().cpu(),
                "mid_scale": mid_scale.detach().cpu(),
            })
            del images, large_tokens, mid_tokens, patch_tokens, class_tokens
            del large_scale, mid_scale
        return batches

    @torch.no_grad()
    def evaluate_support(self, support_image):
        """Evaluate one support image without retaining prior support features."""
        large_memory = mid_memory = patch_memory = None
        try:
            large_memory, mid_memory, patch_memory = memory(
                self.model, [self.class_name], self.dataset_dir, "", self.preprocess,
                self.transform, 1, self.args.few_shot_features, self.dataset_name,
                self.device, specific_image_paths=[support_image],
            )
            gt_masks, anomaly_maps, gt_sp, pr_sp = [], [], [], []

            for batch in self.test_batches:
                gt_masks.append(batch["masks"])
                gt_sp.extend(batch["anomaly"])

                batch_size = batch["batch_size"]
                height, width = batch["height"], batch["width"]
                # Similarity helpers normalize tokens in place, so each support
                # must receive independent copies of the cached raw features.
                large_tokens = batch["large_tokens"].to(self.device, copy=True)
                mid_tokens = batch["mid_tokens"].to(self.device, copy=True)
                patch_tokens = batch["patch_tokens"].to(self.device, copy=True)
                class_tokens = batch["class_tokens"].to(self.device, copy=True)
                large_scale = batch["large_scale"].to(self.device, copy=True)
                mid_scale = batch["mid_scale"].to(self.device, copy=True)
                class_names = [self.class_name] * batch_size
                large_few = harmonic_aggregation(
                    (batch_size, height // self.patch_size, width // self.patch_size),
                    few_shot(large_memory, large_tokens, class_names), large_scale,
                ).cuda()
                mid_few = harmonic_aggregation(
                    (batch_size, height // self.patch_size, width // self.patch_size),
                    few_shot(mid_memory, mid_tokens, class_names), mid_scale,
                ).cuda()
                patch_few = few_shot(patch_memory, patch_tokens, class_names).reshape(
                    batch_size, height // self.patch_size, width // self.patch_size
                ).cuda()
                few_score = torch.nan_to_num(
                    (large_few + mid_few + patch_few) / 3.0, nan=0.0, posinf=0.0, neginf=0.0
                )

                normal_features = self.normal_text[0].expand(batch_size, -1, -1)
                anomaly_features = self.anomaly_text[0].expand(batch_size, -1, -1)
                text_features = torch.cat((normal_features, anomaly_features), dim=1).permute(0, 2, 1)
                image_score = compute_score(class_tokens, text_features)[:, 0, 1]
                large_text = compute_sim(large_tokens, text_features)[:, :, 1]
                mid_text = compute_sim(mid_tokens, text_features)[:, :, 1]
                large_score = harmonic_aggregation(
                    (batch_size, height // self.patch_size, width // self.patch_size), large_text, large_scale
                )
                mid_score = harmonic_aggregation(
                    (batch_size, height // self.patch_size, width // self.patch_size), mid_text, mid_scale
                )
                score_map = torch.nan_to_num(
                    3.0 / (
                        1.0 / large_score.cuda() + 1.0 / mid_score.cuda()
                        + 1.0 / image_score.unsqueeze(1).unsqueeze(1)
                    ),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                score_map = score_map + few_score.cuda()
                image_score = (image_score + torch.amax(few_score, dim=(1, 2))) / 2.0
                score_map = F.interpolate(score_map.unsqueeze(1), size=(height, width), mode="bilinear").squeeze(1)
                pr_sp.extend(image_score.detach().cpu())
                anomaly_maps.append(score_map.detach().cpu())

                # Keep only CPU metric inputs before moving to the next test batch.
                del large_tokens, mid_tokens, patch_tokens, class_tokens
                del large_scale, mid_scale, large_few, mid_few, patch_few, few_score
                del normal_features, anomaly_features, text_features, image_score
                del large_text, mid_text, large_score, mid_score, score_map

            gt_px = torch.cat(gt_masks).squeeze(1).numpy()
            pr_px = torch.cat(anomaly_maps).numpy()
            gt_sp = np.asarray(gt_sp)
            pr_sp = np.asarray(pr_sp)
            px_precision, px_recall, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
            px_f1 = (2 * px_precision * px_recall) / (px_precision + px_recall)
            auroc_sp, f1_sp, ap_sp = safe_image_metrics(gt_sp, pr_sp)
            return {
                "auroc_px": roc_auc_score(gt_px.ravel(), pr_px.ravel()),
                "f1_px": np.max(px_f1[np.isfinite(px_f1)]),
                "ap_px": average_precision_score(gt_px.ravel(), pr_px.ravel()),
                "aupro": cal_pro_score(gt_px, pr_px),
                "auroc_sp": auroc_sp,
                "f1_sp": f1_sp,
                "ap_sp": ap_sp,
            }
        finally:
            del large_memory, mid_memory, patch_memory
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()


@torch.no_grad()
def test(args,):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    target_class = args.class_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    device = "cuda" if torch.cuda.is_available() else "cpu"


    import logging
     # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    model = CLIP_AD(args.model)
    model.to(device)

    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    
    preprocess = model.preprocess

    preprocess.transforms[0] = transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(img_size, img_size))
    # ---------- 数据加载（支持单类过滤）----------
    if dataset_name == 'mvtec':
        all_obj_list = [
            'carpet', 'bottle', 'hazelnut', 'leather', 'cable',
            'capsule', 'grid', 'pill', 'transistor', 'metal_nut',
            'screw', 'toothbrush', 'zipper', 'tile', 'wood'
        ]
    elif dataset_name == 'visa':
        all_obj_list = [
            'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
            'pcb4', 'pipe_fryum'
        ]

    if target_class is not None:
        # 自定义测试数据集（只包含目标类）
        if target_class not in all_obj_list:
            raise ValueError(f"Class {target_class} not found in dataset.")

        with open(os.path.join(dataset_dir, 'meta.json'), 'r') as f:
            meta = json.load(f)
        test_samples = meta['test'][target_class]

        # 定义临时 Dataset 类
        class CustomTestDataset(torch.utils.data.Dataset):
            def __init__(self, samples, root, transform, target_transform):
                self.samples = samples
                self.root = root
                self.transform = transform
                self.target_transform = target_transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                info = self.samples[idx]
                img_path = os.path.join(self.root, info['img_path'])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                mask_path = info['mask_path']
                if mask_path:
                    mask = Image.open(os.path.join(self.root, mask_path))
                    mask = self.target_transform(mask)
                else:
                    mask = torch.zeros((1, img.shape[1], img.shape[2]))
                return {
                    'img': img,
                    'img_mask': mask,
                    'cls_name': target_class,
                    'cls_id': 0,  # 因为只有一类，索引固定为 0
                    'anomaly': info['anomaly']
                }

        test_data = CustomTestDataset(test_samples, dataset_dir, preprocess, transform)
        obj_list = [target_class]
    else:
        # 原有逻辑：加载所有类别
        if dataset_name == 'mvtec':
            test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                     aug_rate=-1, mode='test')
        elif dataset_name == 'visa':
            test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                    mode='test')
        obj_list = all_obj_list
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)


    model.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    patch_size = 16
  
    Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features = prepare_text_future(model, obj_list)
    ########################################
    if args.k_shot == 0:
        few = False
    else:
        few = True

    if few:
        support_paths = [args.support_image] if args.support_image is not None else None
        large_memory, mid_memory, patch_memory = memory(
            model.to(device), obj_list, dataset_dir, save_path, preprocess, transform,
            args.k_shot, few_shot_features, dataset_name, device,
            specific_image_paths=support_paths
        )

    if args.support_image is not None:
        with open(os.path.join(save_path, 'support_image.txt'), 'w') as f:
            f.write(f"Support image: {args.support_image}\n")

    for index, items  in enumerate(tqdm(test_dataloader)):


        images = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        if target_class is not None:
            cls_id = torch.zeros_like(cls_id)  # 因为 obj_list 只有目标类，索引必须为 0
        results['cls_names'].extend(cls_name)
        gt_mask = items['img_mask']
        gt_mask = (gt_mask > 0.5).to(gt_mask.dtype)
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].extend(items['anomaly'].detach().cpu())

        
              
        b, c, h, w = images.shape
  

        average_normal_features = Mermory_avg_normal_text_features[cls_id]
        average_anomaly_features = Mermory_avg_abnormal_text_features[cls_id]
  
      
      
        large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(images, patch_size)
        
        if few:
            m_l = few_shot(large_memory, large_scale_tokens, cls_name)
            m_m = few_shot(mid_memory, mid_scale_tokens, cls_name)
            m_p = few_shot(patch_memory, patch_tokens, cls_name)
        

           
            m_l  =  harmonic_aggregation((b, h//patch_size, w//patch_size) ,m_l, large_scale).cuda()
            m_m  =  harmonic_aggregation((b, h//patch_size, w//patch_size) ,m_m, mid_scale).cuda()
            m_p  =  m_p.reshape((b, h//patch_size, w//patch_size)).cuda()


            few_shot_score = torch.nan_to_num((m_l + m_m + m_p)/3.0, nan=0.0, posinf=0.0, neginf=0.0)

 
        zscore = compute_score(class_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))

        z0score = zscore[:,0,1]


        large_scale_simmarity = compute_sim(large_scale_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))[:,:,1]
        mid_scale_simmarity = compute_sim(mid_scale_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))[:,:,1]

        #####################################multi-scale
        large_scale_score = harmonic_aggregation((b, h//patch_size, w//patch_size) ,large_scale_simmarity, large_scale)
        mid_scale_score  = harmonic_aggregation((b, h//patch_size, w//patch_size), mid_scale_simmarity, mid_scale)
 
        multiscale_score = mid_scale_score

        multiscale_score = torch.nan_to_num(3.0/(1.0/large_scale_score.cuda() + 1.0/mid_scale_score.cuda() + 1.0/z0score.unsqueeze(1).unsqueeze(1)), nan=0.0, posinf=0.0, neginf=0.0)

        ########################################################

        multiscale_score = multiscale_score.cuda().unsqueeze(1)  # Add batch and channel dimensions

        if few:

            multiscale_score = multiscale_score + few_shot_score.cuda().unsqueeze(1)

            z0score = (z0score+ torch.max(torch.max(few_shot_score, dim = 1)[0],dim = 1)[0])/2.0

        multiscale_score = F.interpolate(multiscale_score, size=(h, w), mode='bilinear')

        multiscale_score = multiscale_score.squeeze()
        results['pr_sp'].extend(z0score.detach().cpu())
        results['anomaly_maps'].append(multiscale_score)



    results['imgs_masks'] = torch.cat(results['imgs_masks'])

    results['anomaly_maps'] = torch.cat(results['anomaly_maps']).detach().cpu().numpy()

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)


        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        # print("precisions recalls", precisions, recalls)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        # print("precisions recalls", precisions, recalls)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)


    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    # parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    # few shot
    parser.add_argument("--k_shot", type=int, default=10, help="10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--class_name", type=str, default=None, help="Only test this class (e.g., 'bottle')")
    parser.add_argument("--support_image", type=str, default=None,
                        help="Path to a specific support image (for 1-shot). If provided, k_shot must be 1 and this image will be used as the only support.")
    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)
