# Data and checkpoints

## Dataset

1. Request access on the [dataset page](https://huggingface.co/datasets/shdragron/RoboGeo).
2. Download it (843 GB in total):

   ```bash
   hf auth login
   hf download shdragron/RoboGeo --repo-type dataset --local-dir robogeo_download
   ```

   | part | size | needed for |
   |---|---|---|
   | `metadata/` | 1.5 GB | everything |
   | `platform_layouts/` | 273 GB | everything (Sedan, SUV and Bus images) |
   | `viewpoint_perturbation/` | 568 GB | viewpoint perturbation |
   | `bev_segmentation_labels/` | 1.0 GB | BEV segmentation |

   To skip a part, list the ones you need, e.g.
   `--include "metadata/*" "platform_layouts/*" "bev_segmentation_labels/*"`.

3. Extract into one folder and point `ROBOGEO_DATA` to it:

   ```bash
   export ROBOGEO_DATA=/path/to/robogeo
   cd robogeo_download
   mkdir -p $ROBOGEO_DATA/{carla_geobev,carla_VR,carla_geobev_labels}

   tar -I zstd -xf metadata/carla_geobev_meta.tar.zst -C $ROBOGEO_DATA/carla_geobev
   for t in platform_layouts/*.tar; do tar -xf $t -C $ROBOGEO_DATA/carla_geobev; done
   cp viewpoint_perturbation/viewpoint_metadata.json $ROBOGEO_DATA/carla_VR/
   for t in viewpoint_perturbation/*.tar; do tar -xf $t -C $ROBOGEO_DATA/carla_VR; done
   for t in bev_segmentation_labels/*.tar; do tar -xf $t -C $ROBOGEO_DATA/carla_geobev_labels; done
   ```

   ```
   $ROBOGEO_DATA/
   ├── carla_geobev/          # annotations, splits, info files, sweeps/RGB-*/ (Sedan, SUV, Bus images)
   ├── carla_VR/              # viewpoint_metadata.json, sweeps/RGB-<CAM>_yaw*pitch*roll*/
   └── carla_geobev_labels/   # {sedan,suv,bus}/ (training), {sedan,suv,bus}_eval/ (evaluation)
   ```

4. The detection info files use the mmdet3d path `data/nuscenes/...`. Link the data into each
   detection folder:

   ```bash
   for m in detection/bevformer detection/cape; do
       mkdir -p $m/data && ln -sfn $ROBOGEO_DATA/carla_geobev $m/data/nuscenes
   done
   ```

## Checkpoints

```bash
hf download shdragron/RoboGeo-checkpoints --local-dir checkpoints
export ROBOGEO_CKPT=$PWD/checkpoints
```

```
$ROBOGEO_CKPT/
├── detection/
│   ├── bevformer_tiny/   bevformer_tiny_{sedan,suv,bus}.pth
│   └── cape/             cape_{sedan,suv,bus}.pth
└── segmentation/
    ├── gaussianlss/      gaussianlss_{sedan,suv,bus}.pth
    └── cvt/              cvt_{sedan,suv,bus}.pth
```

One checkpoint per platform. The Sedan checkpoint is evaluated under both protocols. The SUV
and Bus checkpoints give P<sub>oracle</sub> for cross-platform transfer.
