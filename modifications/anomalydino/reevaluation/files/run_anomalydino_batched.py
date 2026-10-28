import os

from src.detection import augment_image
from src.backbones import get_model
from src.utils import get_dataset_info

from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import yaml
from scipy.ndimage import gaussian_filter
from random import sample
from  matplotlib.colors import LinearSegmentedColormap

from argparse import ArgumentParser
from unified_evaluation import (
    InferenceBenchmark,
    compute_metrics,
    format_result,
    make_result_row,
    write_results_csv,
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection")
    parser.add_argument("--model_size", type=str, default="s")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--preprocess", type=str, default="masking_only")
    parser.add_argument("--save_examples", default=True)
    parser.add_argument("--device", default='cuda:3')
    
    args = parser.parse_args()
    return args

def dists_to_score(dists):
    # mean top 1% = empirical tail value at risk (for 99% quantile)
    return np.mean(sorted(dists, reverse = True)[:int(len(dists) * 0.01)])

def calculate_cosine_distances(features_all, sample_index, device, quantile = 0.001):
    """
    Calculate the cosine distances on patch level between the sample with index sample_index and all other samples in the list of features_all.
    The distance of a test patch to all reference patches is calculated as 1 - cosine_similarity,
    then the mean of the closest 0.1% of patches is returned as the score for this test patch. 
    """
    with torch.no_grad():
        features_all_tensors = [torch.tensor(features, device=device, dtype=torch.float32) for features in features_all]
        sample_features = features_all_tensors[sample_index]

        all_features = torch.cat([features for i, features in enumerate(features_all_tensors) if i != sample_index])

        normalized_sample = torch.nn.functional.normalize(sample_features, dim=1)
        normalized_all = torch.nn.functional.normalize(all_features, dim=1)
        cosine_similarity = torch.mm(normalized_sample, normalized_all.t())
        cosine_distance = 1 - cosine_similarity

        # delete some tensors to free up GPU memory (for VisA memory consumption is quite high)
        del cosine_similarity, normalized_sample, normalized_all, all_features, features_all_tensors, sample_features
        # torch.cuda.empty_cache()

        # Calculate the mean (per patch) af all distances below the 1 percent quantile
        quantile_distances = torch.kthvalue(cosine_distance, int(quantile * cosine_distance.shape[1]), dim=1).values.unsqueeze(1)
        mask_lowest = cosine_distance < quantile_distances
        means_below_quantile = (cosine_distance * mask_lowest).sum(dim=1) / mask_lowest.sum(dim=1)
    return means_below_quantile.cpu().numpy()


def evaluate_ad_batched(model,
                        dataset,
                        data_root, 
                        plots_dir,
                        masking_default,
                        device,
                        save_examples = True):
    unified_rows = []

    for object_name in tqdm(objects, position=0, leave=True, desc="Evaluating objects"):

        type_anomalies = object_anomalies[object_name]
        type_anomalies.append('good')

        # ensure that each type is only evaluated once
        type_anomalies = list(set(type_anomalies))

        # order type anomalies alphabetically, but with 'good' at the front
        type_anomalies = sorted(type_anomalies, key=lambda x: (x != 'good', x))

        # Extract reference features
        imgs_all = []
        features_all = []
        masks_ref = []
        gt_label = []
        gt_masks = []
        grid_sizes = []
        benchmark = InferenceBenchmark(device)

        img_test_folder = f"{data_root}/{object_name}/test/"

        with torch.inference_mode():
            # read in all test samples (with label information for later evaluation)
            for type_anomaly in type_anomalies:
                data_dir = img_test_folder + f"{type_anomaly}"
                for img_test_nr in tqdm(sorted(os.listdir(data_dir)), desc=f"Load test set: {object_name} ({type_anomaly})", leave=False):
                    img_test_path = f"{data_dir}/{img_test_nr}"
                    img_test = cv2.cvtColor(cv2.imread(img_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    benchmark.start_batch()
                    image_tensor, grid_size = model.prepare_image(img_test)
                    features = model.extract_features(image_tensor)
                    mask_test = model.compute_background_mask(features, grid_size, threshold=10, masking_type=masking_default[object_name])
                    benchmark.stop_batch(1)
                    imgs_all.append(img_test)
                    features_all.append(features)
                    masks_ref.append(mask_test)
                    gt_label.append(type_anomaly)
                    grid_sizes.append(grid_size)
                    if type_anomaly == "good":
                        gt_mask = np.zeros(img_test.shape[:2], dtype=np.uint8)
                    else:
                        stem = os.path.splitext(img_test_nr)[0]
                        suffix = "_mask.png" if dataset == "MVTec" else ".png"
                        gt_path = f"{data_root}/{object_name}/ground_truth/{type_anomaly}/{stem}{suffix}"
                        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if gt_mask is None:
                            raise FileNotFoundError(f"Ground-truth mask not found: {gt_path}")
                        gt_mask = cv2.resize(
                            gt_mask,
                            (img_test.shape[1], img_test.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    gt_masks.append((gt_mask > 0).astype(np.uint8))

            # score test samples (mutual scoring)
            test_dists = []
            anomaly_maps = []
            y_scores = []
            for test_idx in tqdm(range(len(features_all)), desc = f"Score test set: {object_name}", leave=False):
                benchmark.start_batch()
                dists = calculate_cosine_distances(features_all, test_idx, device=device)
                dists[~masks_ref[test_idx]] = 0.0
                test_dists.append(dists)
                y_scores.append(dists_to_score(dists))
                anomaly_map = cv2.resize(
                    dists.reshape(grid_sizes[test_idx]),
                    (imgs_all[test_idx].shape[1], imgs_all[test_idx].shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                anomaly_maps.append(gaussian_filter(anomaly_map, sigma=4))
                benchmark.stop_batch(0)
            
            y_true = [0 if l == "good" else 1 for l in gt_label]
            metrics = compute_metrics(y_true, y_scores, gt_masks, anomaly_maps)
            row = make_result_row(object_name, metrics, benchmark.result())
            unified_rows.append(row)
            print(format_result(row))

            if save_examples:
                # plot 5 random samples for each category
                sample_indices = []
                for type_anomaly in type_anomalies:
                    indices = [i for i, l in enumerate(gt_label) if l == type_anomaly]
                    sample_indices.extend(sample(indices, 5))

                # plot anomaly map
                y_scores_array = np.asarray(y_scores)
                vmax = np.max(y_scores_array[np.asarray(y_true) == 0]) * 1.2
                # print(vmax)
                fig, axes = plt.subplots(len(type_anomalies), 5, figsize=(10, 2*len(type_anomalies)))
                for i, sample_idx in enumerate(sample_indices):
                    ax = axes[i//5, i%5]
                    ax.imshow(imgs_all[sample_idx])
                    d = anomaly_maps[sample_idx]
                    ax.imshow(d, cmap = cmap, vmax=vmax)

                    ax.set_title(f"{gt_label[sample_idx]}: {dists_to_score(test_dists[sample_idx]):.2f}")
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(f"{plots_dir}/{object_name}_examples.png")
                # plt.show()
                plt.close()

            # empty CUDA cache
            torch.cuda.empty_cache()
            
    return unified_rows

if __name__=="__main__":
    args = parse_args()

    # set torch device
    torch.cuda.set_device(args.device)

    args.model_name = "dinov2_vit" + args.model_size.lower() + "14"
    model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)
    dataset = args.dataset
    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, args.preprocess)

    # colors for anomaly map
    neon_violet = (0.5, 0.1, 0.5, 0.4)
    neon_yellow = (0.8, 1.0, 0.02, 0.7)
    colors = [(1.0, 1, 1.0, 0.0),  neon_violet, neon_yellow]
    cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)

    plot_dir = f"results_{args.dataset}/{args.model_name}_{args.resolution}/batched-0-shot_{args.preprocess}/"
    
    os.makedirs(plot_dir, exist_ok = True)

    print("Results will be saved to", plot_dir)

    # save args to yaml
    with open(f"{plot_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    
    unified_rows = evaluate_ad_batched(model,
                                 args.dataset,
                                 args.data_root, 
                                 plot_dir, save_examples = True, 
                                 device = args.device,
                                 masking_default = masking_default)
    
    benchmark_file = write_results_csv(
        f"{plot_dir}/unified_results.csv", unified_rows
    )
    print(f"Wrote unified benchmark to {benchmark_file}")
