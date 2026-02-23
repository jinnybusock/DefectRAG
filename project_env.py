# 서로 다른 폴더에 있는 DINOv2 & SAM3 불러오기

import sys
import os

def initialize_project():
    # 윈도우 절대 경로 (r을 붙여 이스케이프 문자 오류 방지)
    dinov2_path = r"C:\Users\hjchung\Desktop\dinov2"
    sam3_path = r"C:\Users\hjchung\Desktop\sam3"

    paths = [dinov2_path, sam3_path]

    for p in paths:
        if os.path.exists(p):
            if p not in sys.path:
                sys.path.append(p)
            # 패키지 인식을 위한 __init__.py 자동 생성 (보완점 반영)
            init_file = os.path.join(p, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    pass
                print(f"Created: {init_file}")
        else:
            print(f"Warning: Path not found -> {p}")

    # Triton Mock 설정 (윈도우 환경 필수)
    from unittest.mock import MagicMock
    import importlib.machinery

    mock_spec = importlib.machinery.ModuleSpec('triton', None)
    mock_triton = MagicMock()
    mock_triton.__spec__ = mock_spec
    mock_triton.language = MagicMock()
    mock_triton.language.__spec__ = importlib.machinery.ModuleSpec('triton.language', None)

    sys.modules['triton'] = mock_triton
    sys.modules['triton.language'] = mock_triton.language

    print("Project environment initialized.")


if __name__ == "__main__":
    initialize_project()