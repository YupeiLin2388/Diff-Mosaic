# Diff-Mosaic: Augmenting Realistic Representations in Infrared Small Target Detection via Diffusion Prior
## Inastall

```bash
conda create -n diffmosaic python=3.9
conda activate diffmosaic
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda install xformers==0.0.16 -c xformers
pip install -r requirements.txt
```

## pre-trained model

1. Open clip model:
2. Pixel-prior model:
3. Diff-prior model:

## Data Preparation
1. download the NUDT-SIRST dataset and SIRST dataset
2. run mosaic.py to generate Mosaic image
3. run `degrade.py` to get mix image

## Generate  augmentation sample

```bash
python inference.py --input ./add_noise/NUDT_mosaic/  --config configs/model/diff_prior.yaml --ckpt weights/NUDT_stage2/last.ckpt --swinir_ckpt weights/NUDT_stage1/last.ckpt --steps 50 --sr_scale 1 --repeat_times 1 --color_fix_type wavelet --output results/nudt_moc/ --device cuda --use_guidance --g_scale 400 --g_t_start 200
```

