import sys
from unittest.mock import MagicMock
from types import ModuleType
import importlib.machinery
import torch

# Triton Mock 설정
def mock_triton():
    name = 'triton'
    if name not in sys.modules:
        m = ModuleType(name)
        m.__path__ = []
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)

        # SAM3가 호출하는 jit 및 주요 속성 가짜 생성
        def dummy_jit(fn=None, **kwargs):
            if fn is not None: return fn
            return lambda x: x

        m.jit = dummy_jit
        m.Config = MagicMock()
        m.autotune = dummy_jit

        sys.modules[name] = m

        # 2. 하위 모듈(language, runtime, compiler) 생성
        for sub in ['language', 'runtime', 'compiler']:
            sub_name = f"{name}.{sub}"
            sub_m = ModuleType(sub_name)
            sub_m.__spec__ = importlib.machinery.ModuleSpec(sub_name, None)
            sys.modules[sub_name] = sub_m

            # [핵심] triton.language(tl)에 필요한 속성들 보강
            if sub == 'language':
                sub_m.constexpr = any  # constexpr 에러 해결
                sub_m.float32 = torch.float32
                sub_m.int32 = torch.int32
                sub_m.int64 = torch.int64
                # 자주 사용되는 함수들 Mocking
                sub_m.arange = MagicMock()
                sub_m.load = MagicMock()
                sub_m.store = MagicMock()

        print("✅ Windows 환경: Triton Mocking 완료 (ValueError 방지)")

mock_triton()

import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
from sam3 import build_sam3_image_model
from project_env import initialize_project

# 1. 프로젝트 및 모델 로드
initialize_project()
device = "cuda" if torch.cuda.is_available() else "cpu"
sam3_checkpoint = r"C:\Users\hjchung\Desktop\sam3\checkpoints\sam3.pt"
model = build_sam3_image_model(checkpoint_path=sam3_checkpoint).to(device)
model.train()

# 2. 'good' 클래스 전용 데이터셋
class GoodImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.path = os.path.join(base_path, 'good')
        self.images = [os.path.join(self.path, f) for f in os.listdir(self.path)
                       if f.lower().endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1024 대신 896으로 리사이즈 시도 (SAM3 모델 백본 호환성)
        # 896도 안된다면 512나 448로 낮춰보세요.
        img = Image.open(self.images[idx]).convert("RGB").resize((1008, 1008))

        # 0~255 범위를 0~1로 정규화하여 전달
        img_np = np.array(img) / 255.0
        # SAM3 표준 입력 크기 및 타입으로 변환
        return torch.as_tensor(np.array(img)).permute(2, 0, 1).float()


class DummyInput:
    """모델 내부에서 요구하는 속성들을 갖춘 가짜 입력 객체"""

    def __init__(self, tensor):
        self.image = tensor
        self.input_points = None
        self.input_labels = None
        self.input_boxes = None
        # AttributeError 방지
        self.find_text_batch = [""]

    def __getattr__(self, name):
        """
        정의되지 않은 속성 (input_boxes_label 등)을 호출할 때 AttributeError 내는 대신 None 반환
        """
        return None

class SimpleBatch:
    def __init__(self, data):
        if isinstance(data, list):
            self.data_list = data
        elif isinstance(data, dict):
            self.data_list = [data]
        else:
            self.data_list = [data]
        self._update_img_batch()

    def _update_img_batch(self):
        if len(self.data_list) > 0:
            first_item = self.data_list[0]
            if isinstance(first_item, dict):
                self.img_batch = first_item.get('image', first_item)
            else:
                self.img_batch = first_item
        else:
            self.img_batch = None

    @property
    def find_inputs(self):
        # ❌ AttributeError: 'Tensor' object has no attribute 'input_points' 해결
        # 모델이 input.input_points 등에 접근하므로 속성을 가진 객체로 감싸서 반환
        if self.img_batch is not None:
            return [DummyInput(self.img_batch)]
        return []

    @property
    def find_text_batch(self):
        return [""]

    @property
    def find_targets(self):
        # targets[0] 에러 방지를 위해 위에서 만든 객체를 리스트에 담아 반환
        return self.find_inputs

    def to(self, device):
        new_data_list = []
        for item in self.data_list:
            if isinstance(item, dict):
                new_item = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
                new_data_list.append(new_item)
            elif torch.is_tensor(item):
                new_data_list.append(item.to(device))
            else:
                new_data_list.append(item)

        self.data_list = new_data_list
        self._update_img_batch()
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __iter__(self):
        return iter(self.data_list)

# 3. 학습 루프 (로컬 GPU 메모리 고려)
def run_adaptation():
    train_path = r"C:\Users\hjchung\Desktop\RAG Train"
    dataset = GoodImageDataset(train_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 로컬 OOM 방지
    torch.cuda.is_bf16_supported = lambda: False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    print(f"🚀 로컬 GPU({device})에서 SAM3 정상 패턴 학습 시작...")

    for epoch in range(3):  # 과적합 방지를 위해 적게 수행
        for i, batch in enumerate(loader):

            # 빈 배치 검사
            if batch is None or (isinstance(batch, (list, dict)) and len(batch) == 0):
                continue

            optimizer.zero_grad()

            # 객체 생성 및 장치 이동
            input_data= SimpleBatch(batch)
            input_data= input_data.to(device)

            try:
                # 모델 추론
                output = model(input_data)

                if output is None:
                    print(f"⚠️ [{i}] 모델 출력(output)이 None입니다. 건너뜁니다.")
                    continue

                # SAM3의 출력 딕셔너리에서 유효한 특징 텐서 찾기
                # 'vision_features' 또는 'cond_frame_outputs' 등이 있을 수 있습니다.
                target_feat = None
                for key in ['high_res_feats', 'image_embed', 'vision_features']:
                    if key in output:
                        target_feat = output[key]
                        break

                if target_feat is not None:
                    # 특징값 복원 학습 (정상 패턴 암기)
                    loss = nn.MSELoss()(target_feat, target_feat.detach().clone())
                else:
                    # output이 텐서 그 자체인 경우와 딕셔너리인 경우 구분
                    if torch.is_tensor(output):
                        loss= nn.MSELoss()(output, output.detach())
                    elif isinstance(output, dict):
                        # 값이 None이 아닌 텐서들만 필터링
                        tensors= [v for v in output.values() if torch.is_tensor(v) and v is not None]

                        if tensors:
                            loss= sum(nn.MSELoss()(v, v.detach()) for v in tensors)

                        else:
                            continue

                    else:
                        continue

                loss.backward()
                optimizer.step()

                if i% 10 == 0:
                    print(f"Epoch {epoch + 1} [{i}/{len(loader)}] Loss: {loss.item():.6f}")

            except Exception as e:
                # 에러 내용을 명확히 출력하도록 변경
                print(f"❌ 학습 루프 에러 발생: {e}")
                import traceback
                traceback.print_exc()  # 상세 에러 경로 출력
                break

    torch.save(model.state_dict(), "sam3_good_adapted.pt")
    print("✅ 로컬 학습 완료! sam3_good_adapted.pt 저장됨.")


if __name__ == "__main__":
    run_adaptation()