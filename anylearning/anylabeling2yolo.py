import os
import shutil
import math
import pathlib
import random
from collections import OrderedDict

import json
import cv2


class AnyLabeling2YOLO(object):

    def __init__(self, json_dir, to_seg=False):
        self._json_dir = json_dir
        self._label_id_map = self._get_label_id_map(self._json_dir)
        self._to_seg = to_seg

    def _get_label_id_map(self, json_dir):
        label_set = set()

        for file_name in os.listdir(json_dir):
            if file_name.endswith("json"):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data["shapes"]:
                    label_set.add(shape["label"])

        return OrderedDict(
            [(label, label_id) for label_id, label in enumerate(label_set)]
        )

    def _train_val_test_split(self, json_names, val_size=0.0, test_size=0.0):
        if val_size == 0.0 and test_size == 0.0:
            return json_names, []

        num_samples = len(json_names)
        num_test_samples = int(num_samples * test_size)
        num_val_samples = int(num_samples * val_size)
        num_train_samples = num_samples - num_val_samples - num_test_samples

        random.seed(42)
        train_samples = random.sample(json_names, num_train_samples)
        json_names = [name for name in json_names if name not in train_samples]
        val_samples = random.sample(json_names, num_val_samples)
        test_samples = [name for name in json_names if name not in val_samples]
        return train_samples, val_samples, test_samples


    def convert(self, output_dir, val_size=0.0, test_size=0.0):
        json_names = [
            file_name
            for file_name in os.listdir(self._json_dir)
            if os.path.isfile(os.path.join(self._json_dir, file_name))
            and file_name.endswith(".json")
        ]

        assert val_size + test_size < 1.0, "val_size + test_size should be less than 1.0"
        train_json_names, val_json_names, test_json_names = self._train_val_test_split(
            json_names, val_size=val_size, test_size=test_size
        )

        # Convert anylabeling object to yolo format object, and save them to files
        # also get image from anylabeling json file and save them under images folder
        for subset_dir, json_names in zip(
            ("train/", "val/", "test/"), (train_json_names, val_json_names, test_json_names)
        ):
            if len(json_names) == 0:
                continue
            target_dir = os.path.join(output_dir, subset_dir)
            labels_target_dir = os.path.join(target_dir, "labels")
            images_target_dir = os.path.join(target_dir, "images")
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(images_target_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(labels_target_dir).mkdir(parents=True, exist_ok=True)
            for json_name in json_names:
                print(f"Processing {json_name} ...")
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path))
                img_path = self._copy_image(
                    json_data, json_name, self._json_dir, images_target_dir
                )
                yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
                self._save_yolo_label(
                    json_name, labels_target_dir, yolo_obj_list
                )

        print("Generating dataset.yaml file ...")
        self._save_dataset_yaml(output_dir)

    def _copy_image(
        self, json_data, json_name, image_dir_path, target_dir
    ):
        img_name = json_data["imagePath"].split("/")[-1]
        src_img_path = os.path.join(image_dir_path, img_name)
        dst_img_path = os.path.join(target_dir, img_name)
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_img_path, dst_img_path)
        return dst_img_path

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []

        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data["shapes"]:
            if shape["shape_type"] == "circle":
                yolo_obj = self._get_circle_shape_yolo_object(
                    shape, img_h, img_w
                )
            else:
                yolo_obj = self._get_other_shape_yolo_object(
                    shape, img_h, img_w
                )

            yolo_obj_list.append(yolo_obj)

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape["label"]]
        obj_center_x, obj_center_y = shape["points"][0]

        radius = math.sqrt(
            (obj_center_x - shape["points"][1][0]) ** 2
            + (obj_center_y - shape["points"][1][1]) ** 2
        )

        if self._to_seg:
            retval = [label_id]

            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1

            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [
                [
                    obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                    obj_center_y - math.sin(i * math.pi / n_part2) * radius,
                ]
                for i in range(1, n_part)
            ]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()

            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])

            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape["label"]]

        if self._to_seg:
            retval = [label_id]
            for i in shape["points"]:
                i[0] = round(float(i[0]) / img_w, 6)
                i[1] = round(float(i[1]) / img_h, 6)
                retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)

            x_lists = [port[0] for port in obj_port_list]
            y_lists = [port[1] for port in obj_port_list]

            return (
                min(x_lists),
                __get_dist(x_lists),
                min(y_lists),
                __get_dist(y_lists),
            )

        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape["points"])

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _save_yolo_label(
        self, json_name, target_dir, yolo_obj_list
    ):
        txt_path = os.path.join(
            target_dir, json_name.replace(".json", ".txt")
        )

        with open(txt_path, "w+") as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f"{i} "
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += "\n"
                f.write(yolo_obj_line)

    def _save_dataset_yaml(self, output_dir):
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        yaml_content = ""

        if os.path.exists(os.path.join(output_dir, "train/")):
            yaml_content += "train: %s\n" % "train/"
        if os.path.exists(os.path.join(output_dir, "val/")):
            yaml_content += "val: %s\n" % "val/"
        if os.path.exists(os.path.join(output_dir, "test/")):
            yaml_content += "test: %s\n" % "test/"

        yaml_content += "nc: %i\n" % len(self._label_id_map)

        names_str = ""
        for label, _ in self._label_id_map.items():
            names_str += "'%s', " % label
        names_str = names_str.rstrip(", ")
        yaml_content += "names: [%s]" % names_str

        with open(yaml_path, "w+") as yaml_file:
            yaml_file.write(yaml_content)

        print(f"\nDataset yaml file is saved to {yaml_path}")
        print(yaml_content)
