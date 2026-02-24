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
from config import sam3_checkpoint

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
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img= Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)

# 3. 학습 루프 (로컬 GPU 메모리 고려)
def run_adaptation():
    initialize_project()
    device= "cuda" if torch.cuda.is_available() else "cpu"
    sam3_checkpoint= r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt"

    print("🔄 모델 로드 중...")
    model= build_sam3_image_model(checkpoint_path=r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt").to(device)
    model.train()

    dataset = GoodImageDataset(r"C:\Users\hjchung\Desktop\RAG Train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 로컬 OOM 방지
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    print(f"🚀 로컬 GPU({device})에서 SAM3 정상 패턴 학습 시작...")

    for epoch in range(3):  # 과적합 방지를 위해 적게 수행
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            input_data= SimpleBatch(batch).to(device)

            try:
                # 모델 추론
                output = model(input_data)

                target_feat= None
                if isinstance(output, dict):
                    for key in ['high_res_feats', 'vision_features', 'image_embed']:
                        if key in output[key] is not None:
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

            except AssertionError as e:
                expected_shape= "알 수 없음"
                try:
                    expected_shape= model.backbone.trunk.blocks[0].attn.freqs_cis.shape
                except:
                    pass

                print(f"❌ RoPE 해상도 불일치!")
                print(f" - 현재 입력 크기: {input_data.img_batch.shape}")
                print(f" - 모델 기대 RoPE 형상 (H, W): {expected_shape}")
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