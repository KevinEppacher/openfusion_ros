# Reducing GPU Memory Usage in OpenFusion

OpenFusion uses two main parameters:

- **voxel_size**: metric size of one voxel  
- **block_resolution**: number of voxels per block  

Physical block size is:

```
block_size_m = voxel_size * block_resolution
```

If you increase `voxel_size`, you **must decrease `block_resolution`** to keep block size stable.

---

## Example

Old:
```
voxel_size: 0.01953125
block_resolution: 8
block_size = 0.15625 m
```

New (4× larger voxels):
```
voxel_size: 0.078125
block_resolution: 8   # WRONG → block_size = 0.625 m (unstable)
```

Correct:
```
voxel_size: 0.078125
block_resolution: 2   # keeps block_size ≈ 0.15625 m
```

---

## Recommended Settings (Matterport-scale)

**Stable**
```
voxel_size: 0.078125
block_resolution: 2
```

**More aggressive**
```
block_resolution: 3
```

---

## TSDF Setting

Scale truncation with voxel size:
```
tsdf_trunc = voxel_size * 3
```

---

## Rule of Thumb

**If voxel size increases by N× → block resolution must decrease by N×.**


# Saving Semantic Maps in OpenFusion ROS

```bash
ros2 service call /openfusion_ros/run_semantic_map openfusion_msgs/srv/SaveSemanticMap "
dataset_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac
scene_name: 00800-TEEsavR23oF
version: v1.1
semantic_class_list_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac/00800-TEEsavR23oF/assets/classes.json
overwrite: true
"
```

**Scene name: 00813-svBbv1Pavdk**
```bash
ros2 service call /openfusion_ros/run_semantic_map openfusion_msgs/srv/SaveSemanticMap "
dataset_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac
scene_name: 00813-svBbv1Pavdk
version: v1.1
semantic_class_list_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac/00813-svBbv1Pavdk/assets/classes.json
overwrite: true
"
```

**Scene name: 00814-p53SfW6mjZe**
```bash
ros2 service call /openfusion_ros/run_semantic_map openfusion_msgs/srv/SaveSemanticMap "
dataset_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac
scene_name: 00814-p53SfW6mjZe
version: v1.4
semantic_class_list_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac/00814-p53SfW6mjZe/assets/classes.json
overwrite: true
"
```

**Scene name: 00824-Dd4bFSTQ8gi**
```bash
ros2 service call /openfusion_ros/run_semantic_map openfusion_msgs/srv/SaveSemanticMap "
dataset_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac
scene_name: 00824-Dd4bFSTQ8gi
version: v1.0
semantic_class_list_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac/00824-Dd4bFSTQ8gi/assets/classes.json
overwrite: true
"
```

**Scene name: 00848-ziup5kvtCCR**
```bash
ros2 service call /openfusion_ros/run_semantic_map openfusion_msgs/srv/SaveSemanticMap "
dataset_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac
scene_name: 00848-ziup5kvtCCR
semantic_class_list_path: /app/src/sage_evaluator/sage_datasets/matterport_isaac/00848-ziup5kvtCCR/assets/classes.json
version: v1.0
overwrite: true
"
```