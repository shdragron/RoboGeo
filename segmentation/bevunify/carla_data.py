"""RoboGeo BEV segmentation data: per-scene datasets and the sample transform.

Each sample is one entry of ``<labels_dir>/<scene>.json``. The transform loads the six RGB
images (resized and top-cropped to 224x480), the intrinsics, the ego-to-camera extrinsics
and the pre-rendered BEV labels. Two hooks change the camera geometry for robustness
evaluation:

- viewpoint perturbation (``set_viewpoint``): for the perturbed cameras, IMG loads the
  rotated renders from carla_VR, EXT applies the rotation to the extrinsics only, and CAL
  does both;
- cross-platform transfer (``cts_ext_override``): maps the extrinsics of the target
  platform to the Sedan extrinsics through a per-camera rig delta (see ``sedan_rig_delta``).
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pyquaternion import Quaternion


def read_split(dataset_dir, split):
    """Scene names listed in ``<dataset_dir>/split/<split>.txt``."""
    return (Path(dataset_dir) / "split" / f"{split}.txt").read_text().strip().split("\n")


def subset_tokens(dataset_dir, platform, subset):
    """Sample tokens of an evaluation subset; None means all evaluation frames.

    ``fps16`` is the 768-frame sample list of ``<platform>_infos_val_fps16.pkl`` (16 frames
    per scene), meant for quick checks only.
    """
    if subset == "full":
        return None
    if subset != "fps16":
        raise ValueError(f"unknown subset {subset!r} (expected 'full' or 'fps16')")
    with open(Path(dataset_dir) / f"{platform}_infos_val_fps16.pkl", "rb") as f:
        return {info["token"] for info in pickle.load(f)["infos"]}


def decode(img, n):
    """Bit-packed BEV label image -> (h, w, n) array of {0, 1}."""
    shift = np.arange(n, dtype=np.int32)[None, None]
    x = np.array(img)[..., None]
    return (x >> shift) & 1


def _ego_from_cam(pose):
    R = Quaternion(pose["sensor2ego_rotation"]).rotation_matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pose["sensor2ego_translation"])
    return T


def _cam_from_ego(pose):
    R = Quaternion(pose["sensor2ego_rotation"]).rotation_matrix
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(pose["sensor2ego_translation"])
    return T


class LoadDataTransform:
    def __init__(self, dataset_dir, labels_dir, image, num_classes, label_indices,
                 split_intrin_extrin=False, gt_center=True, gt_offset=True, gt_visibility=True,
                 vr_root=None, viewpoint_metadata_path=None):
        self.dataset_dir = Path(dataset_dir)
        self.labels_dir = Path(labels_dir)
        self.image_config = dict(h=int(image["h"]), w=int(image["w"]), top_crop=int(image["top_crop"]))
        self.num_classes = num_classes
        self.label_indices = [[int(i) for i in group] for group in label_indices]
        self.split_intrin_extrin = split_intrin_extrin
        self.gt_center = gt_center
        self.gt_offset = gt_offset
        self.gt_visibility = gt_visibility
        self.to_tensor = torchvision.transforms.ToTensor()

        # Viewpoint perturbation. The carla_VR root comes from the config; the 'vr_root'
        # field stored in viewpoint_metadata.json is ignored.
        self.vr_root = Path(vr_root) if vr_root else None
        self.viewpoint_metadata_path = viewpoint_metadata_path
        self.viewpoint_metadata = None
        self.vp_variant = None              # e.g. 'yaw0pitch-12roll0'; None = unperturbed
        self.vp_image_swap = False
        self.vp_extrinsic_swap = False
        self.vp_cameras = None              # perturbed cameras; None = all six

        # Cross-platform transfer: {camera: 4x4 rig delta}, applied as E <- D @ E.
        self.cts_ext_override = None

        # Optional evaluation caches: token -> labels, image path -> (tensor, width, height).
        # Only unperturbed images are cached.
        self.gt_cache = None
        self.image_cache = None

    def set_viewpoint(self, variant, image_swap, extrinsic_swap, cameras=None):
        if variant is not None and self.viewpoint_metadata is None:
            if self.vr_root is None or self.viewpoint_metadata_path is None:
                raise ValueError("viewpoint perturbation needs data.vr_root and data.viewpoint_metadata_path")
            self.viewpoint_metadata = json.loads(Path(self.viewpoint_metadata_path).read_text())
        self.vp_variant = variant
        self.vp_image_swap = bool(image_swap)
        self.vp_extrinsic_swap = bool(extrinsic_swap)
        self.vp_cameras = None if cameras is None else set(cameras)

    def _perturbed(self, camera):
        return self.vp_variant is not None and (self.vp_cameras is None or camera in self.vp_cameras)

    def _vr_image_path(self, image_path, camera):
        """carla_VR render of the same scene and time: geobev frame N is carla_VR frame 2N."""
        v = self.vp_variant
        m = re.search(r"scene-(\d+)-frame-(\d+)", Path(image_path).name)
        scene, frame = int(m.group(1)), int(m.group(2))
        name = f"SimBEV-scene-{scene:04d}-frame-{2 * frame:04d}-RGB-{camera}_{v}.jpg"
        return self.vr_root / "sweeps" / f"RGB-{camera}_{v}" / name

    def _vr_extrinsic_correction(self, scene, camera):
        """C such that E_variant = C @ E_base (E maps ego to camera)."""
        poses = self.viewpoint_metadata["scenes"][scene][camera]
        return _cam_from_ego(poses[self.vp_variant]) @ _ego_from_cam(poses["yaw0pitch0roll0"])

    def get_cameras(self, sample):
        h, w, top_crop = self.image_config["h"], self.image_config["w"], self.image_config["top_crop"]
        h_resize, w_resize = h + top_crop, w
        cameras = sample["cam_channels"]

        images, intrinsics = [], []
        for image_path, I_original, camera in zip(sample["images"], sample["intrinsics"], cameras):
            swap = self._perturbed(camera) and self.vp_image_swap
            path = self._vr_image_path(image_path, camera) if swap else self.dataset_dir / image_path
            cacheable = self.image_cache is not None and not swap
            if cacheable and str(path) in self.image_cache:
                img, image_w, image_h = self.image_cache[str(path)]
            else:
                image = Image.open(path)
                resized = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
                img = self.to_tensor(resized.crop((0, top_crop, resized.width, resized.height)))
                image_w, image_h = image.width, image.height
                if cacheable:
                    self.image_cache[str(path)] = (img, image_w, image_h)

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image_w
            I[0, 2] *= w_resize / image_w
            I[1, 1] *= h_resize / image_h
            I[1, 2] *= h_resize / image_h
            I[1, 2] -= top_crop

            images.append(img)
            intrinsics.append(torch.tensor(I))

        extrinsics = torch.tensor(np.float32(sample["extrinsics"]))
        if self.vp_variant is not None and self.vp_extrinsic_swap:
            for k, camera in enumerate(cameras):
                if self._perturbed(camera):
                    C = self._vr_extrinsic_correction(sample["scene"], camera)
                    extrinsics[k] = torch.tensor(C.astype(np.float32)) @ extrinsics[k]
        if self.cts_ext_override is not None:
            for k, camera in enumerate(cameras):
                D = torch.tensor(np.float32(self.cts_ext_override[camera]))
                extrinsics[k] = D @ extrinsics[k]

        lidar2img = []
        for I, E in zip(intrinsics, extrinsics):
            K = torch.eye(4, dtype=torch.float32)
            K[:3, :3] = I
            lidar2img.append(K @ E)

        result = {"image": torch.stack(images, 0), "lidar2img": torch.stack(lidar2img, 0)}
        if self.split_intrin_extrin:
            result["intrinsics"] = torch.stack(intrinsics, 0)
            result["extrinsics"] = extrinsics
        return result

    def get_bev(self, sample):
        if self.gt_cache is not None and sample["token"] in self.gt_cache:
            return self.gt_cache[sample["token"]]

        scene_dir = self.labels_dir / sample["scene"]
        bev = decode(Image.open(scene_dir / sample["bev"]), self.num_classes)
        bev = self.to_tensor((255 * bev).astype(np.uint8))           # (num_classes, h, w) in [0, 1]
        bev = torch.stack([bev[group].amax(dim=0) for group in self.label_indices], 0)
        result = {"vehicle": bev}

        if self.gt_visibility:
            visibility = Image.open(scene_dir / sample["visibility"])
            result["vehicle_visibility"] = torch.from_numpy(np.array(visibility, dtype=np.uint8))
        if self.gt_center or self.gt_offset:
            aux = np.load(scene_dir / sample["aux"])["aux"]
            if self.gt_center:
                result["vehicle_center"] = self.to_tensor(aux[..., 1])
            if self.gt_offset:
                result["vehicle_offset"] = torch.tensor(aux[..., 2:4]).permute(2, 0, 1).float()

        if self.gt_cache is not None:
            self.gt_cache[sample["token"]] = result
        return result

    def __call__(self, sample):
        result = self.get_cameras(sample)
        result.update(self.get_bev(sample))
        result["token"] = sample["token"]
        result["scene"] = sample["scene"]
        return result


class SceneDataset(torch.utils.data.Dataset):
    """Samples of one scene. Unreadable files raise instead of being skipped."""

    def __init__(self, labels_dir, scene, transform, tokens=None):
        samples = json.loads((Path(labels_dir) / f"{scene}.json").read_text())
        if tokens is not None:
            samples = [s for s in samples if s["token"] in tokens]
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.transform(self.samples[idx])


def build_datasets(data_cfg, split, labels_dir=None, tokens=None):
    """One dataset per scene of ``split``, all sharing one transform.

    ``labels_dir`` defaults to the platform's training labels for 'train' and to its
    evaluation labels for 'val'.
    """
    if labels_dir is None:
        labels_dir = data_cfg.labels_dir if split == "train" else data_cfg.val_labels_dir
    transform = LoadDataTransform(
        data_cfg.dataset_dir, labels_dir, data_cfg.image, data_cfg.num_classes,
        data_cfg.label_indices, split_intrin_extrin=data_cfg.split_intrin_extrin,
        gt_center=data_cfg.gt_center, gt_offset=data_cfg.gt_offset,
        gt_visibility=data_cfg.gt_visibility, vr_root=data_cfg.vr_root,
        viewpoint_metadata_path=data_cfg.viewpoint_metadata_path)
    scenes = read_split(data_cfg.dataset_dir, split)
    return [SceneDataset(labels_dir, scene, transform, tokens) for scene in scenes]


def sedan_rig_delta(sedan_labels_dir, target_labels_dir):
    """Per-camera rig delta D = E_sedan @ inv(E_target) from the first evaluation sample.

    The platforms drive the same trajectory, so the per-frame ego pitch and roll cancel and
    D @ E_target(t) gives the Sedan extrinsics at every frame t.
    """
    def first_sample(labels_dir):
        scene_file = sorted(Path(labels_dir).glob("scene_*.json"))[0]
        sample = json.loads(scene_file.read_text())[0]
        return {cam: np.array(E, dtype=np.float64)
                for cam, E in zip(sample["cam_channels"], sample["extrinsics"])}

    sedan, target = first_sample(sedan_labels_dir), first_sample(target_labels_dir)
    return {cam: (sedan[cam] @ np.linalg.inv(target[cam])).astype(np.float32) for cam in target}
