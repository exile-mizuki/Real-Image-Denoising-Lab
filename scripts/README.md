# Experimental scripts

- `nind`: original native NIND manifest + downloader + fixed-patch experiments.
- `polyu_sidd_renoir`: paired LQ/GT and RENOIR visual evaluation. Uses `--sidd-scenes` and `--renoir-cameras` to select groups.
- `dnd`: HDF5 sRGB + official bounding boxes, no public GT-based scores.

All BAT launchers in this repository use **repository-relative** data directories at `datasets/<name>`, and write to `outputs/<name>` (both ignored by Git). You may override the dataset directory by setting `DENOISE_DATA_DIR` to the full dataset folder for that one run.

Python source code was taken from the **latest tested versions** used in the course project. Existing historical CSV results are not regenerated when packaging this repository. Performance numbers reflect the original runtime and hardware, not a browser measurement.
