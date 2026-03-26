"""
model_loader.py
===============
Chịu trách nhiệm DUY NHẤT: tải model về local và khởi tạo LlamaCpp.

Quy trình:
  1. Kiểm tra xem file GGUF đã tồn tại local chưa
  2. Nếu chưa → download từ HuggingFace Hub (có progress bar)
  3. Khởi tạo LlamaCpp với đầy đủ InputParams
  4. Trả về instance LLM sẵn sàng để inference

Tách module này ra để:
  - Dễ swap model mà không động vào logic chat
  - Có thể cache/reuse instance LLM
  - Unit test độc lập
"""

import sys
import time
from pathlib import Path
from typing import Optional

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from config import (
    ModelConfig,
    InputParams,
    InferenceParams,
    OutputParams,
    model_config,
    input_params,
    inference_params,
    output_params,
)


# ---------------------------------------------------------------------------
# DOWNLOAD MODEL
# ---------------------------------------------------------------------------

def download_model(
    cfg: ModelConfig = model_config,
) -> Path:
    """
    Tải file GGUF từ HuggingFace Hub về local nếu chưa tồn tại.

    Sử dụng huggingface_hub.hf_hub_download — có resume support,
    không download lại nếu file đã có và hash khớp.

    Returns:
        Path: đường dẫn local tới file model đã tải về
    """
    # Tạo thư mục models nếu chưa có
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra xem model đã tồn tại local chưa
    if cfg.model_path.exists():
        size_mb = cfg.model_path.stat().st_size / (1024 * 1024)
        print(f"[model_loader] ✅ Model đã có tại local: {cfg.model_path}")
        print(f"[model_loader]    Kích thước: {size_mb:.1f} MB")
        return cfg.model_path

    # Chưa có → tiến hành download
    print(f"[model_loader] ⬇️  Bắt đầu download model: {cfg.hf_filename}")
    print(f"[model_loader]    Nguồn: https://huggingface.co/{cfg.hf_repo}")
    print(f"[model_loader]    Lưu tại: {cfg.model_path}")
    print(f"[model_loader]    (TinyLlama Q4_K_M ≈ 670MB, vui lòng chờ...)\n")

    try:
        from huggingface_hub import hf_hub_download

        start_time = time.time()
        local_path = hf_hub_download(
            repo_id=cfg.hf_repo,
            filename=cfg.hf_filename,
            local_dir=str(cfg.model_dir),
            local_dir_use_symlinks=False,   # Copy thực, không dùng symlink
        )
        elapsed = time.time() - start_time

        print(f"\n[model_loader] ✅ Download hoàn tất sau {elapsed:.1f}s")
        print(f"[model_loader]    Lưu tại: {local_path}")
        return Path(local_path)

    except ImportError:
        print("[model_loader] ❌ Thiếu thư viện: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"[model_loader] ❌ Download thất bại: {e}")
        print("[model_loader]    Thử download thủ công và đặt vào ./models/")
        sys.exit(1)


# ---------------------------------------------------------------------------
# KHỞI TẠO LLM
# ---------------------------------------------------------------------------

def build_llm(
    model_path: Optional[Path] = None,
    in_params: InputParams = input_params,
    inf_params: InferenceParams = inference_params,
    out_params: OutputParams = output_params,
) -> LlamaCpp:
    """
    Khởi tạo instance LlamaCpp với đầy đủ tham số từ config.

    LangChain LlamaCpp nhận tất cả tham số (input + inference + output)
    trong constructor — model được load vào RAM/VRAM tại đây.

    Args:
        model_path: override đường dẫn model (mặc định dùng model_config)
        in_params:  InputParams — tài nguyên phần cứng
        inf_params: InferenceParams — điều khiển sinh token
        out_params: OutputParams — cách trả kết quả

    Returns:
        LlamaCpp instance đã sẵn sàng inference
    """
    if model_path is None:
        model_path = model_config.model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {model_path}\n"
            f"Hãy chạy download_model() trước."
        )

    print(f"[model_loader] 🔧 Đang khởi tạo LlamaCpp...")
    print(f"[model_loader]    Model : {model_path.name}")
    print(f"[model_loader]    n_ctx : {in_params.n_ctx} tokens")
    print(f"[model_loader]    n_threads : {in_params.n_threads}")
    print(f"[model_loader]    n_gpu_layers: {in_params.n_gpu_layers}")
    print(f"[model_loader]    temperature : {inf_params.temperature}")
    print(f"[model_loader]    max_tokens  : {inf_params.max_tokens}")

    # Callbacks để stream token ra stdout trong chế độ CLI
    callbacks = []
    if out_params.streaming:
        callbacks.append(StreamingStdOutCallbackHandler())

    start_time = time.time()

    llm = LlamaCpp(
        # ── Model path ──────────────────────────────────────────────────────
        model_path=str(model_path),

        # ── INPUT PARAMS: tài nguyên phần cứng ──────────────────────────────
        n_ctx=in_params.n_ctx,              # Context window size
        n_threads=in_params.n_threads,      # CPU threads
        n_gpu_layers=in_params.n_gpu_layers,# GPU layers (0 = CPU only)
        n_batch=in_params.n_batch,          # Decode batch size

        # ── INFERENCE PARAMS: điều khiển sinh token ──────────────────────────
        temperature=inf_params.temperature, # Độ sáng tạo
        max_tokens=inf_params.max_tokens,   # Max output tokens
        top_p=inf_params.top_p,             # Nucleus sampling
        top_k=inf_params.top_k,             # Top-K sampling
        repeat_penalty=inf_params.repeat_penalty,  # Chống lặp từ
        stop=inf_params.stop_sequences,     # Stop sequences

        # ── OUTPUT PARAMS: cách trả kết quả ─────────────────────────────────
        streaming=out_params.streaming,     # Stream token về ngay
        echo=out_params.echo,               # Không echo lại prompt

        # ── MISC ────────────────────────────────────────────────────────────
        verbose=in_params.verbose,          # Tắt llama.cpp logs
        callbacks=callbacks,
    )

    elapsed = time.time() - start_time
    print(f"[model_loader] ✅ LlamaCpp sẵn sàng! (load trong {elapsed:.2f}s)\n")

    return llm


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION — dùng trong các module khác
# ---------------------------------------------------------------------------

def load_model() -> LlamaCpp:
    """
    Hàm tiện ích: download (nếu cần) + khởi tạo LLM trong một bước.
    Đây là entry point chính mà các module khác nên gọi.

    Returns:
        LlamaCpp instance đã sẵn sàng
    """
    model_path = download_model()
    return build_llm(model_path=model_path)


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test model_loader standalone ===\n")
    llm = load_model()
    print("\n--- Test inference nhanh ---")
    response = llm.invoke("Say 'Hello World' in one sentence.")
    print(f"\nResponse: {response}")
