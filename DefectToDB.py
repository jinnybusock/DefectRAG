import os
import torch
import numpy as np
import psycopg2
from PIL import Image
from pgvector.psycopg2 import register_vector
import cv2
from project_env import initialize_project
from sam3 import build_sam3_image_model
from config import sam3_checkpoint
import torch.nn.functional as F
import math

# 1. 환경 초기화 및 모델 로드
initialize_project()
device = "cuda" if torch.cuda.is_available() else "cpu"

# SAM3 로드 (학습된 가중치 적용)
model_sam3 = build_sam3_image_model(checkpoint_path=sam3_checkpoint).to(device)

# 학습시킨 가중치 로드
adapted_path= "sam3_good_adapted.pt"
if os.path.exists(adapted_path):
    # weights_only= True는 보안을 위해 권장
    model_sam3.load_state_dict(torch.load(adapted_path, map_location=device, weights_only=True))
    print(f"✅ 학습된 가중치 로드 완료: {adapted_path}")

else:
    print(f"⚠️ 가중치 파일을 찾을 수 없어 베이스 모델로 진행합니다: {adapted_path}")

model_sam3.eval()

# DINOv2 로드
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model_dinov2.eval()


class SimpleBatch:
    def __init__(self, img_batch):
        self.img_batch = img_batch

        # SAM3 모델의 num_frames== 1 검사 통과하기 위해 요소 하나를 넣음
        self.find_inputs= [None]

    def to(self, device):
        self.img_batch = self.img_batch.to(device)
        return self

def get_bbox_from_mask(mask_tensor, threshold=0.5):
    """SAM3 출력 마스크에서 결함 부위의 Bounding Box 좌표 추출"""
    mask = (mask_tensor > threshold).cpu().numpy().astype(np.uint8)[0, 0]  # [H, W]
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return (x, y, w, h)
    return None

def extract_dinov2_feature(img_patch):
    """잘라낸 결함 영역에서 DINOv2 벡터 추출"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img_patch).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model_dinov2(img_t)
    return features.cpu().numpy().flatten()

def process_and_save_to_db(folder_path, defect_label):
    # DB 연결
    conn = psycopg2.connect(host="localhost", dbname="DefectRAGUpdate", user="postgres", password="3510")
    register_vector(conn)
    cur = conn.cursor()

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
    saved_count= 0     # 저장된 개수 세기 위한 변수

    print(f"\n--- [{defect_label}] 데이터 구축 시작 ---")

    for img_name in image_files:
        full_path = os.path.join(folder_path, img_name)
        raw_image = Image.open(full_path).convert("RGB")

        # SAM3용 입력 전처리
        input_img_res = raw_image.resize((1008, 1008))

        # tensor 변환 및 정규화
        img_np= np.array(input_img_res) / 255.0     # 0~1 범위로 정규화
        input_tensor = torch.as_tensor(np.array(input_img_res)).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # SAM3 모델 요구 사양에 맞춘 강제 리사이징 전처리
        target_res= 1024

        for m in model_sam3.modules():
            if hasattr(m, 'freqs_cis') and m.freqs_cis is not None:
                num_patches = m.freqs_cis.shape[0]
                side= int(math.sqrt(num_patches))
                target_res= side* 16
                break

        if input_tensor.shape[-1] != target_res:
            print(f" [전처리] 이미지 크기를 모델 최적 해상도({target_res})로 변경합니다.")

            input_tensor= F.interpolate(
                input_tensor,
                size=(target_res, target_res),
                mode="bilinear",
                align_corners=False
            )

        # SimpleBatch 객체 생성 및 모델 추론
        batch_data= SimpleBatch(input_tensor)

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ 알림: [{defect_label}] 폴더 처리 완료!")
    print(f"📊 요약: 총 {len(image_files)}개 중 {saved_count}개의 데이터가 DB에 정상 저장되었습니다.")
    print(f"--------------------------------------\n")

if __name__ == "__main__":
    # 각 defect별 폴더 경로 리스트
    defects = [
        ("crack", r"C:\Users\hjchung\Desktop\RAG Train\crack"),
        ("fabDefect", r"C:\Users\hjchung\Desktop\RAG Train\fabDefect"),
        ("ink", r"C:\Users\hjchung\Desktop\RAG Train\ink"),
        ("mapout", r"C:\Users\hjchung\Desktop\RAG Train\mapout"),
        ("particle", r"C:\Users\hjchung\Desktop\RAG Train\particle"),
        ("unknown", r"C:\Users\hjchung\Desktop\RAG Train\unknown")
    ]

    for label, path in defects:
        try:
            process_and_save_to_db(path, label)
        except Exception as e:
            print(f"❌ 에러 발생 [{label}]: {e}")

            import traceback
            traceback.print_exc()
            # 에러가 발생해도 다음 폴더로 넘어가도록 처리
            continue

    print("🎉 모든 결함 데이터베이스 구축 공정이 최종 완료되었습니다!")