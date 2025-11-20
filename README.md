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
