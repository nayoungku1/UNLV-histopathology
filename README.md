# Histopathological Image Analysis for Tumor Detection @UNLV
UNLV EIP Summer 2025 Project: pathological image analysis for cancer classification

* **Team Member** 🙌: [Nayoung](https://github.com/nayoungku1), [Yeonseo](https://github.com/yeonseo1129), [Chaemin](https://github.com/twemmi), Amaan
---
* Original Dataset in `.svs`: [Cancer Image Archive](https://pathdb.cancerimagingarchive.net/eaglescope/dist/?configurl=%2Fsystem%2Ffiles%2Fcollectionmetadata%2F202208%2Fbiobank_metadata_page_first50_4.json&filterState=%5B%7B%22id%22%3A%22TCIA_Collection%22%2C%22title%22%3A%22Collection%22%2C%22field%22%3A%22TCIA_Collection%22%2C%22operation%22%3A%22eq%22%2C%22values%22%3A%22CMB-LCA%22%7D%5D)
* Patched Dataset in `.npz`: [HuggingFace](https://huggingface.co/datasets/nayoungku1/npz-histopathology-dataset)

### How to make WSI into patches
```conda run -n aiap python extract_patch_npy.py --input [svs containing dir] --output [dir where npz files should be saved]```


