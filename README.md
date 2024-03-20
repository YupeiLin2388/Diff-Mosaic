# Diff-Mosaic: Augmenting Realistic Representations in Infrared Small Target Detection via Diffusion Prior
## Inastall

```bash
conda create -n diffmosaic python=3.9
conda activate diffmosaic
pip install -r requirements.txt
```

## pre-trained model

1. Download Open clip model( [[huggingface]](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin)) and place it in `./weights/` 
2. Download Pixel-prior model ( [Google Drive]) and place in `./weights/`
3. Download Diff-prior model ( [Google Drive]) and place in `./weights/`

## Data Preparation
1. download the NUDT-SIRST dataset and SIRST dataset
2. run `mosaic.py` to generate Mosaic image
3. run `degrade.py` to get mix image

## Generate  augmentation sample

```bash
python inference.py --input ./add_noise/NUDT_mosaic/  --config configs/model/diff_prior.yaml --ckpt weights/NUDT_stage2/last.ckpt --swinir_ckpt weights/NUDT_stage1/last.ckpt --steps 50 --sr_scale 1 --repeat_times 1 --color_fix_type wavelet --output results/nudt_moc/ --device cuda --use_guidance --g_scale 400 --g_t_start 200
```

