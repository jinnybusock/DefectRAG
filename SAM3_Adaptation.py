import sys
import torch
from types import ModuleType
from unittest.mock import MagicMock
import importlib.util

# Triton Mock 설정
def mock_triton():
    if 'triton' not in sys.modules:
        m= ModuleType('triton')
        m.__spec__= importlib.util.spec_from_loader('triton', loader= None)
        m.jit= lambda f= None, **k: (f if f else lambda x: x)
        m.autotune= m.jit
        m.Config= MagicMock()
        sys.modules['triton'] = m

        for s in ['language', 'runtime', 'compiler']:
            s_name= f"{'triton'}.{s}"
            sm= ModuleType(s_name)
            sm.__spec__= importlib.util.spec_from_loader(s_name, loader= None)

            if s== 'language':
                sm.constexpr= lambda x: x
                sm.float32, sm.int32, sm.int64= torch.float32, torch.int32, torch.int64
                sm.arange, sm.exp= torch.arange, torch.exp
                sm.load, sm.store, sm.dot= MagicMock(), MagicMock(), MagicMock()
                sm.max_contiguous= lambda x, y: x
            sys.modules[f'triton.{s}'] = sm

mock_triton()

import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from sam3 import build_sam3_image_model
from project_env import initialize_project
from config import bpe_path
from config import sam3_checkpoint
import sam3.model.vitdet as vitdet

target_size= 2224

def preprocess_image(img_tensor, targetSize= (target_size, target_size)):
    """
    어떤 크기의 이미지가 들어오든 모델이 원하는 target_size로 강제 고정
    img_tensor: [B, C, H, W] 형태
    """
    # 현재 크기 확인
    curr_h, curr_w = img_tensor.shape[-2:]

    # 이미 타겟 사이즈라면 그대로 반환
    if(curr_h, curr_w) == targetSize:
        return img_tensor
    print(f" [전처리] 이미지 크기 변경: ({curr_h}, {curr_w}) -> {targetSize}")

    # 모델이 허용하는 크기로 강제 리사이징
    # SAM3는 정사각 입력 선호 -> align_corners는 False가 일반적
    return F.interpolate(img_tensor, size= targetSize, mode="bilinear", align_corners=False)

def patched_reshaped_for_broadcast(freqs_cis, x):
    """
    라이브러리의 AssertionError 우회하기 위한 패치
    """
    # 실제 연산에 필요한 형태로 변환 (1, L, 1, D)
    ndim= x.ndim
    assert ndim>= 2
    # ndim-2 (시퀀스 길이)와 ndim-1 (헤드 차원)을 유지
    shape= [d if i in (ndim- 2, ndim- 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# 라이브러리 함수를 우리가 만든 패치 함수로 교체
vitdet.reshape_for_broadcast= patched_reshaped_for_broadcast

class DummyInput:
    """모델 내부에서 요구하는 속성들을 갖춘 가짜 입력 객체"""

    def __init__(self, tensor):
        self.image = tensor
        self.input_points = None
        self.input_labels = None
        self.input_boxes = None

    def __getattr__(self, name):
        """
        정의되지 않은 속성 (input_boxes_label 등)을 호출할 때 AttributeError 내는 대신 None 반환
        """
        return None

class SimpleBatch:
    def __init__(self, tensor):
        # 모델이 직접 참조하는 img_batch 설정
        self.img_batch= tensor
        self.dummy_item= DummyInput(tensor)

    @property
    def find_inputs(self):
        return [self.dummy_item]

    @property
    def find_text_batch(self):
        return [""]

    @property
    def find_targets(self):
        return [self.dummy_item]

    def __iter__(self):
        return iter([self.dummy_item])
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return self.dummy_item

    def to(self, device):
        self.img_batch = self.img_batch.to(device)
        self.dummy_item.image= self.img_batch
        return self

# 데이터셋 수정 (표준 전처리 적용)
class GoodImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.path = os.path.join(base_path, 'good')
        self.images= [os.path.join(self.path, f) for f in os.listdir(self.path)
                      if f.lower().endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img= Image.open(self.images[idx]).convert("RGB")
        # 강제로 크기 고정
        img= img.resize((target_size, target_size), Image.BILINEAR)
        return self.transform(img)

# 3. 학습 루프 (로컬 GPU 메모리 고려)
def run_adaptation():
    initialize_project()
    device= "cuda" if torch.cuda.is_available() else "cpu"

    print("🔄 모델 로드 중...")

    model= None     # 초기값 설정
    try:
        # 일반적인 SAM2/3 빌더 형식 (체크포인트 경로만 전달)
        model= build_sam3_image_model(
            bpe_path= bpe_path,
            checkpoint_path= sam3_checkpoint
        ).to(device)
        print("✅ 모델 로드 성공")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return     # 모델 로드 실패 시 함수 종료 (이후 코드 실행 방지)

    model.train()

    dataset = GoodImageDataset(r"C:\Users\hjchung\Desktop\RAG Train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 로컬 OOM 방지
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    print(f"🚀 로컬 GPU({device})에서 SAM3 정상 패턴 학습 시작...")

    for epoch in range(3):  # 과적합 방지를 위해 적게 수행
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            # 데이터를 모델과 동일한 장치로 이동
            input_data= SimpleBatch(batch).to(device)
            # 이미지 크기 고정 전처리 과정
            input_data.img_batch= preprocess_image(input_data.img_batch, targetSize= (target_size, target_size))
            # 메모리 포맷과 데이터 타입을 모델의 backbone에 맞춤
            input_data.img_batch= input_data.img_batch.to(memory_format=torch.contiguous_format)

            # print(f"DEBUG: img_batch shape: {input_data.img_batch.shape}")

            target_res= 2528

            if input_data.img_batch.shape[-1:]!= target_size:
                print(f" [전처리] 모델 요구 사양에 맞춰 {input_data.img_batch.shape[-2:]} -> ({target_res}, {target_res})로 강제 고정합니다.")

                input_data.img_batch= F.interpolate(
                    input_data.img_batch,
                    size= (target_res, target_res),
                    mode="bilinear",
                    align_corners=False
                )

            try:
                # 자동 혼합 정밀도(Autocast) 사용 (SAM3 내부 RoPE 연산 오류 방지)
                with torch.amp.autocast('cuda', dtype= torch.bfloat16):
                    # 모델 추론
                    output = model(input_data)

                target_feat= None
                if isinstance(output, dict):
                    for key in ['high_res_feats', 'vision_features', 'image_embed']:
                        if key in output.get(key) is not None:
                            target_feat = output[key]
                            break

                if target_feat is not None:
                    # 특징값 복원 학습 (정상 패턴 암기)
                    loss = nn.MSELoss()(target_feat, target_feat.detach().clone())
                    loss.backward()
                    optimizer.step()

                    if i% 10==0:
                        print(f"Epoch {epoch+1} [{i}/{len(loader)}] Loss: {loss.item():.6f}")

                    else:
                        print("⚠️ 유효한 특징량을 찾을 수 없습니다.")

            except AssertionError:
                print(f"❌ RoPE 해상도 불일치 발생!")
                import traceback
                traceback.print_exc()

                found= False
                # 모델의 모든 하위 모듈 뒤져서 RoPE 설정값 찾기
                for name, module in model.named_modules():
                    if hasattr(module, 'freqs_cis'):
                        freqs= module.freqs_cis

                        if freqs is not None:
                            print(f" - [발견] 모듈 위치: {name}")
                            print(f" - [발견] RoPE 형상: {freqs.shape}")

                            import math
                            # SAM 계열은 보통 (H*W/256, D) 형태를 가집니다.
                            # 만약 shape[0] 이 4096이면 64x64 그리드 -> 1024x1024 해상도
                            num_patches= freqs.shape[0]
                            side= int(math.sqrt(num_patches))
                            print(f" - 추정 그리드 크기: {side} x {side}")
                            print(f" - 권장 입력 해상도: {side * 16} x {side * 16}")
                            found= True
                            break     # 하나만 찾으면 중단

                if not found:
                    print(" - 모델 내부에서 freqs_cis 속성을 찾을 수 없습니다.")

                print(f" - 현재 입력 텐서 크기: {input_data.img_batch.shape}")
                break

            except Exception as e:
                print(f"❌ 기타 에러 발생: {e}")
                import traceback
                traceback.print_exc()
                break

    torch.save(model.state_dict(), "sam3_good_adapted.pt")
    print("✅ 로컬 학습 완료! sam3_good_adapted.pt 저장됨.")

if __name__ == "__main__":
    run_adaptation()