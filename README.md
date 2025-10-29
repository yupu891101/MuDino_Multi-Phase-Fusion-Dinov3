# DynoDino-Dynamic-Info-Fusion-Dinov3
This repo leverage the multi-phase information of CT image to segment tumor

# initialize setup:
```sh
    uv sync
    uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    uv pip install -e dynodino/external_lib/dino
``` 

LiTS - MSD challenge (https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
KiTS23 (https://github.com/neheller/kits23?tab=readme-ov-file)
AMOS22 (http://www.amos.sribd.cn/download.html)
BTCV (https://www.synapse.org/Synapse:syn3193805/files/)

WAW-TACE (https://zenodo.org/records/12741586)
https://zenodo.org/records/12741586/files/ct_scans_1_4_wawtace_09_05_24.zip?download=1
https://zenodo.org/records/12741586/files/ct_scans_2_4_wawtace_09_05_24.zip?download=1
https://zenodo.org/records/12741586/files/ct_scans_3_4_wawtace_09_05_24.zip?download=1
https://zenodo.org/records/12741586/files/ct_scans_4_4_wawtace_09_05_24.zip?download=1
https://zenodo.org/records/12741586/files/organ_masks_wawtace_09_05_2024.zip?download=1
https://zenodo.org/records/12741586/files/tumor_masks_wawtace_v1_08_05_2024.zip?download=1

Primary Liver Cancer CECT Imaging Dataset (https://www.scidb.cn/en/detail?dataSetId=d685a0b9f8974a2a9d7c880be1dc36e9)
https://china.scidb.cn/download?fileId=cfdd5e37e35a013dc0015aa361531b01&traceId=6a0e7a50-697c-4619-8552-00fbaae6c4ca