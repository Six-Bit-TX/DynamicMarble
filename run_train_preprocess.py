import os
import subprocess
from coarse_init_main import main
from dust3r.model import AsymmetricCroCo3DStereo


GPU_ID = 0
DATA_ROOT_DIR = "/mnt/e/dataset/4drecon/"
DATASETS = ["GTAV"]
SCENES = ["NightSharkOnHighway"]
FRAME = [f"{i:04d}" for i in range(200, 600)]  # Adjust range if needed
N_VIEW = 5
gs_train_iter = 1000
pose_lr = "1x"

model_path = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
device = "cuda"
model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
batch_size = 10
schedule = "linear"
lr = 0.01
niter = 300
n_views = 5
focal_avg=True

for dataset in DATASETS:
    for scene in SCENES:
        for n_frame in FRAME:
            print(f"========= {scene}: Dust3r_coarse_geometric_initialization =========")
            SOURCE_PATH = os.path.join(DATA_ROOT_DIR, dataset, scene, f"frame_{n_frame}")
            print(f"{SOURCE_PATH}")
            img_base_path = SOURCE_PATH
            img_folder_path = os.path.join(img_base_path)
            os.makedirs(img_folder_path, exist_ok=True)
            main(model, img_folder_path, n_views, device=device, batch_size=batch_size, niter=niter, schedule=schedule, lr=lr, focal_avg=focal_avg)