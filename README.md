# Grounded Correspondence

This is the code release for the paper **Rethinking Temporal Consistency in Video Object-Centric Learning: From Prediction to Correspondence (ICML 2026)**, by [Anonymous Authors].

- [Paper](link_to_arxiv)

## Summary

The de facto approach in video object-centric learning maintains temporal consistency through learned dynamics modules that predict future slot states. We demonstrate that these predictors function as expensive approximations of discrete correspondence problems. Modern self-supervised vision backbones already encode instance-discriminative features that distinguish objects reliably. Exploiting these features eliminates the need for learned temporal prediction. 

We introduce **Grounded Correspondence**, a framework that replaces parametric transitioners with deterministic discrete optimization. Slots initialize from saliency peaks in frozen DINOv2 backbone features. Frame-to-frame identity is maintained through Hungarian matching on slot representations. The approach requires **zero learnable parameters** for temporal modeling yet achieves competitive performance on MOVi-D, MOVi-E, and YouTube-VIS.

## Usage

### Setup

First, setup the python environment. We use [Poetry](https://python-poetry.org/):
```bash
poetry install
```

### Install Options

- `poetry install -E tensorflow` to convert tensorflow datasets
- `poetry install -E coco` to use COCO API
- `poetry install -E notebook` to use jupyter notebook and matplotlib

Test the installation:
```bash
poetry run python -m slotcontrast.train tests/configs/test_dummy_image.yml
```

### Data

Follow the instructions in [data/README.md](data/README.md) to download the datasets.
By default, datasets are expected in `./data`. You can change this by setting the environment variable `SLOTCONTRAST_DATA_PATH` or using the `--data-dir` option.

### Training

Train on MOVi-D:
```bash
poetry run python -m slotcontrast.train configs/grounded_correspondence/movi_d.yaml \
    --data-dir ./data \
    --log-dir ./logs
```

Train on MOVi-E:
```bash
poetry run python -m slotcontrast.train configs/grounded_correspondence/movi_e.yaml \
    --data-dir ./data \
    --log-dir ./logs
```

Train on YouTube-VIS 2021:
```bash
poetry run python -m slotcontrast.train configs/grounded_correspondence/ytvis2021.yaml \
    --data-dir ./data \
    --log-dir ./logs
```

To continue training from a checkpoint:
```bash
poetry run python -m slotcontrast.train --continue <path_to_checkpoint> configs/grounded_correspondence/movi_d.yaml
```

### Inference

Run inference on your own videos:
```bash
poetry run python -m slotcontrast.inference --config configs/inference/movi_d_gc.yaml
```

Update `checkpoint: path/to/checkpoint.ckpt` in the config to point to your checkpoint.

For MOVi datasets (visualization only):
```bash
python data/batch_inference.py \
    --checkpoint checkpoints/GC_movid/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/movi_d_gc.yaml \
    --data-dir data/movi_d_raw/valid \
    --output-dir data/inference_results \
    --n-slots 15 \
    --device cuda
```

For YouTube-VIS (with metrics):
```bash
python data/batch_inference.py \
    --checkpoint checkpoints/GC_ytvis/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/ytvis2021_gc.yaml \
    --data-dir data/ytvis2021_raw/valid \
    --output-dir data/inference_results \
    --n-slots 7 \
    --device cuda
```

## Results

We list the results obtained with the configs in this repository:

| Dataset      | Model Variant    | Video FG-ARI | Video mBO | Config                      | Checkpoint Link |
|--------------|------------------|--------------|-----------|-----------------------------|-------------------------------------------------|
| MOVi-D       | ViT-B/14, DINOv2 | 73.7         | 28.4      | grounded_correspondence/movi_d.yaml | [Checkpoint](link) |
| MOVi-E       | ViT-B/14, DINOv2 | 75.7         | 23.4      | grounded_correspondence/movi_e.yaml | [Checkpoint](link) |
| YT-VIS 2021  | ViT-B/14, DINOv2 | 33.1         | 29.3      | grounded_correspondence/ytvis2021.yaml | [Checkpoint](link) |

**Key hyperparameters:**
- **MOVi-D/E**: 15 slots, Grounded Saliency with α=0.5 (D) or α=1.0 (E), spatial radius r=1
- **YouTube-VIS**: 7 slots, Grounded Saliency with α=0.5, spatial radius r=2
- **Temporal**: Hungarian matching (zero learnable parameters)

## Method Overview

**Grounded Saliency Initialization**: Slots are initialized from saliency peaks in frozen DINOv2 features using local-global consistency metric: `S_i = L_i - α·G_i`, where `L_i` measures local instance consistency and `G_i` suppresses background.

**Hungarian Correspondence**: Frame-to-frame identity is maintained through optimal bipartite matching on slot features using the Hungarian algorithm, requiring no learned temporal parameters.

## Citation
```bibtex
@inproceedings{anonymous2026rethinking,
    title={Rethinking Temporal Consistency in Video Object-Centric Learning: From Prediction to Correspondence},
    author={Anonymous Authors},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2026}
}
```

## Acknowledgement

The codebase is adapted from [Videosaur](https://github.com/martius-lab/videosaur) and [SlotContrast](https://github.com/amazon-science/object-centric-learning-framework).

## License

This codebase is released under the MIT license.
Some parts were adapted from other codebases and are governed by their respective licenses.