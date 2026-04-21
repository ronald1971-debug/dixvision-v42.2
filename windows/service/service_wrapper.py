r"""
windows/service/service_wrapper.py
DIX VISION v42.2 — Windows Service Wrapper (NSSM compatible)

Install as service:
  nssm install "DIX_VISION" C:\Python311\python.exe
  nssm set "DIX_VISION" AppDirectory C:\dix_vision_v42_2
  nssm set "DIX_VISION" AppParameters main.py
  nssm start "DIX_VISION"
"""
from __future__ import annotations


def run_as_service() -> None:
    from main import main
    try:
        main()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass

if __name__ == "__main__":
    run_as_service()
