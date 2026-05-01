import json
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import random
import cv2
import random
from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Data_Map
import os
import argparse
import sys

def save_json_append(filepath, new_data):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Se il file non è una lista, trasformalo in lista
    if not isinstance(data, list):
        data = [data]

    # Aggiungi il nuovo dato
    data.append(new_data)

    # Riscrivi il file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def sam_based_count(rgb_image, sam_model):

    img = np.expand_dims(rgb_image, axis=0)

    img = np.squeeze(img)

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = np.ascontiguousarray(img)

    results = sam_model(img)

    annotated = results[0].plot()

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    num_objects = len(results[0].masks.data)
    # print("Numero oggetti segmentati:", num_objects)
    masks = results[0].masks.data
    return num_objects, masks

def build_ellipse(depth, scaling_factor):
    h, w = depth.shape

    cx, cy = w // 2, h // 2
    a = int(scaling_factor * w)
    b = int(scaling_factor * h)
    thickness = 2  # spessore bordo

    y, x = np.ogrid[:h, :w]

    # Ellisse esterna
    outer = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) <= 1

    # Ellisse interna (più piccola)
    inner = ((x - cx)**2 / (a - thickness)**2 + (y - cy)**2 / (b - thickness)**2) <= 1

    # Bordo = differenza
    ellipse_border = outer & (~inner)
    return ellipse_border, inner

def assess_complexity(depth, f_id, ellipse_cache, sam_model=None):
    depth = depth[0,:,:]
    cwd = os.getcwd()
    # point = environment.get_point()
    # rgb_sensor_name = 'SceneV1'
    # rgb_image = environment.grid.get_data_point(point, rgb_sensor_name)

    # source_x, source_y, source_z, source_direction = point.unpack()
    if depth is None:
        raise ValueError("Impossibile caricare l'immagine.")

    pixels = depth.reshape((-1, 1))


    bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=5000).item()

    # print("Estimated Bandwidth:", bandwidth)

    # ---- MEAN SHIFT ----
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(pixels)

    clustered = labels.reshape(depth.shape)

    distance_mapping = {}

    for cluster_id in np.unique(labels):
        cluster_pixels = pixels[labels == cluster_id]
        mean_value = np.mean(cluster_pixels)
        distance_mapping[cluster_id] = mean_value

    depth_draw = depth.copy()
    clustered_draw = clustered.copy()

    # total_objects, masks = sam_based_count(rgb_image, sam_model)

    img_stats = {
            'id': f_id,
            # 'sam_total_objects': total_objects,
            'shift_total_levels': len(labels),
            'mean_depth': float(np.mean(depth)),
            'var_depth': float(np.var(depth)),
            'ellipse_details':dict(),
            'message': 'ok'
        }

    objects_distances = dict()
    for scaling_factor in scaling_factors:

        ellipse_border, inner = ellipse_cache[scaling_factor]
        
        depth_draw[ellipse_border] = np.max(depth)/2

        clustered_draw[ellipse_border] = np.max(clustered)/2

        labels_in_ellipse = clustered[inner]
        unique_clusters = np.unique(labels_in_ellipse)

        distances_per_ellipse = dict()

        for cluster_id in unique_clusters:
            distances_per_ellipse[int(cluster_id)] = float(distance_mapping[cluster_id])

        # for i, mask in enumerate(masks):
        #     # intersezione tra maschera oggetto e ellisse
        #     mask = mask.cpu()
        #     intersection = mask & inner

        #     # percentuale di overlap
        #     overlap_ratio = intersection.sum() / (mask.sum() + 1e-8)

        #     # soglia per dire "oggetto dentro ellisse"
        #     if overlap_ratio > 0:
        #         # if f'object_{i}' in objects_distances.keys():
        #         #     objects_distances[f'object_{i}'] += 1
        #         # else:
        #         #     objects_distances[f'object_{i}'] = 1

        img_stats['ellipse_details'][f'scaling_{scaling_factor}'] = {
                'total_pixels': len(labels_in_ellipse),
                'number_of_clusters': len(unique_clusters),
                'cluster_distances_mapping': distances_per_ellipse,
                # 'objects_count': objects_distances
            }
    
    return img_stats

def main(args):
    map_name = args.map_name
    cwd = os.path.join(args.abs_path, "map_tool_box/DetectionOnTheFlight/01_LighObjCount")
    data_map = Data_Map.DataMapRoof(map_name, memory_saver=False)

    source_x, source_y, source_z, source_direction = 0, 0, 0, 4

    point = Data_Structure.Point(source_x, source_y, source_z, source_direction)

    # instantiate the data_dict that is needed to popilate the data_map.data_dict
    rgb_sensor_name = 'SceneV1'
    depth_sensor_name = 'DepthV1'

    # fetch data
    rgb_image = data_map.get_data_point(point, rgb_sensor_name)
    depth_map = data_map.get_data_point(point, depth_sensor_name)
    
    counter = 0

    for source_x in data_map.data_dicts["SceneV1"].keys():
        for source_y in data_map.data_dicts["SceneV1"][source_x].keys():
            for source_z in data_map.data_dicts["SceneV1"][source_x][source_y].keys():
                for source_direction in data_map.data_dicts["SceneV1"][source_x][source_y][source_z].keys():
                    point = Data_Structure.Point(source_x, source_y, source_z, source_direction)

                    # fetch data
                    rgb_image = data_map.get_data_point(point, rgb_sensor_name)
                    depth_map = data_map.get_data_point(point, depth_sensor_name)

                    depth = depth_map[0]

                    if depth is None:
                        raise ValueError("Impossibile caricare l'immagine.")

                    pixels = depth.reshape((-1, 1))


                    bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=5000)

                    # print("Estimated Bandwidth:", bandwidth)

                    # ---- MEAN SHIFT ----
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    labels = ms.fit_predict(pixels)

                    clustered = labels.reshape(depth.shape)

                    distance_mapping = {}

                    for cluster_id in np.unique(labels):
                        cluster_pixels = pixels[labels == cluster_id]
                        mean_value = np.mean(cluster_pixels)
                        distance_mapping[cluster_id] = mean_value


                    scaling_factors = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                    depth_draw = depth.copy()
                    clustered_draw = clustered.copy()

                    total_objects, masks = sam_based_count(rgb_image, model_path=cwd)

                    img_stats = {
                            'img_metadata': (source_x, source_y, source_z, source_direction),
                            'sam_total_objects': total_objects,
                            'shift_total_levels': len(labels),
                            'mean_depth': float(np.mean(depth)),
                            'var_depth': float(np.var(depth)),
                            'ellipse_details':dict()
                        }

                    objects_distances = dict()
                    for scaling_factor in scaling_factors:

                        ellipse_border, inner = build_ellipse(depth, scaling_factor)
                        
                        depth_draw[ellipse_border] = np.max(depth)/2

                        clustered_draw[ellipse_border] = np.max(clustered)/2

                        labels_in_ellipse = clustered[inner]
                        unique_clusters = np.unique(labels_in_ellipse)

                        distances_per_ellipse = dict()

                        for cluster_id in unique_clusters:
                            distances_per_ellipse[int(cluster_id)] = float(distance_mapping[cluster_id])

                        for i, mask in enumerate(masks):
                            # intersezione tra maschera oggetto e ellisse
                            mask = mask.cpu()
                            intersection = mask & inner

                            # percentuale di overlap
                            overlap_ratio = intersection.sum() / (mask.sum() + 1e-8)

                            # soglia per dire "oggetto dentro ellisse"
                            if overlap_ratio > 0:
                                if f'object_{i}' in objects_distances.keys():
                                    objects_distances[f'object_{i}'] += 1
                                else:
                                    objects_distances[f'object_{i}'] = 1

                        img_stats['ellipse_details'][f'scaling_{scaling_factor}'] = {
                                'total_pixels': len(labels_in_ellipse),
                                'number_of_clusters': len(unique_clusters),
                                'cluster_distances_mapping': distances_per_ellipse,
                                'objects_count': objects_distances
                            }


                    save_json_append(os.path.join(cwd, "results.json"), img_stats)

                    if random.random() < 0.2:
                        fig, ax = plt.subplots(1,2, figsize=(12,5))
                        ax[0].set_title("Original Depth image")
                        ax[0].imshow(depth_draw, cmap='viridis')

                        ax[1].set_title("Mean Shift segmentation")
                        ax[1].imshow(clustered_draw, cmap='viridis')
                        plt.savefig(os.path.join(cwd, f"out_{source_x}_{source_y}_{source_z}_{source_direction}.png"))
                    
                    counter += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Input complexity analysis")

    parser.add_argument("--abs_path", default='./', type=str, help="Root path for data files management")
    parser.add_argument("--map_name", default='AirSimNH', type=str, help="Evaluated map name")

    args = parser.parse_args()
    main(args)
