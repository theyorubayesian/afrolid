import os
import requests
import tarfile
import tempfile
from pathlib import Path
from platformdirs import user_cache_dir
from typing import Any, Final, Optional

import torch
from tqdm import tqdm
from transformers import BatchEncoding, T5Tokenizer

from .conversion import create_pytorch_state_dict
from .language_info import LanguageInfo, Languages
from .model import AfroLIDModel

AFROLID_CACHE_DIR: Final[Path] = Path(os.getenv("AFROLID_CACHE_DIR", user_cache_dir('afrolid')))
AFROLID_CACHE_DIR.mkdir(parents=True, exist_ok=True)

AFROLID_DOWNLOAD_URL: Final[str] = 'https://demos.dlnlp.ai/afrolid/afrolid_model.tar.gz'


def download_and_extract_model(path: Optional[str] = None) -> None:
    if any(
        (Path(p) / "afrolid_model/afrolid_v1_checkpoint.pt").exists()
        for p in [path, AFROLID_CACHE_DIR] if p is not None
    ):
        return
    
    download_dir = Path(path or tempfile.mkdtemp())
    download_path = download_dir / 'afrolid_model.tar.gz'

    response = requests.get(AFROLID_DOWNLOAD_URL, stream=True)
    file_size = int(response.headers.get('content-length', 0))

    with tqdm(total=file_size, unit="iB", unit_scale=True, unit_divisor=1024) as progress_bar:
        with download_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                progress_bar.update(size)
    
    with tarfile.open(str(download_path), 'r:*') as tar:
        tar.extractall(AFROLID_CACHE_DIR if path is None else path)
    
    if not path:
        download_path.unlink()


def load_afrolid_artifacts(download_path: Optional[str] = None) -> tuple[AfroLIDModel, T5Tokenizer, Languages]:
    afrolid = AfroLIDModel()

    model_path = Path(download_path) if download_path else AFROLID_CACHE_DIR

    if (model_path / "torch_model.bin").exists():
        afrolid.load_state_dict(torch.load(model_path / "torch_model.bin", weights_only=False), strict=False)
    else:
        download_and_extract_model(download_path)
        fairseq_dict = torch.load(model_path / "afrolid_model/afrolid_v1_checkpoint.pt", weights_only=False)
        conversion_result = create_pytorch_state_dict(fairseq_dict["model"])

        # TODO: @theyorubayesian - Sanity checks
        torch_dict = conversion_result["new_state_dict"]
        torch.save(torch_dict, model_path / "torch_model.bin")

        afrolid.load_state_dict(torch_dict, strict=False)

    afrolid = afrolid.eval()
    
    tokenizer = T5Tokenizer.from_pretrained(str(model_path / "afrolid_model/afrolid_spm_517_bpe.model"))
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 3
    tokenizer.model_max_length = 1024

    language_info = Languages(model_path / "afrolid_model/dict.label.txt")

    return afrolid, tokenizer, language_info


def prepare_inputs_for_model(text: str | list[str], tokenizer: T5Tokenizer, **encoding_kwargs: Any) -> BatchEncoding:
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, **encoding_kwargs)

    not_eos_mask = (encoding["input_ids"] != tokenizer.eos_token_id)
    combined_mask = encoding["attention_mask"] * not_eos_mask

    # Increment only the non-masked and non-EOS positions by 1
    # FairSeq's encodings always returns 1 greater than the spm Processor's encoding
    encoding["input_ids"] = encoding["input_ids"] + combined_mask
    return encoding


def predict_language(
    text: str | list[str],
    model: AfroLIDModel,
    tokenizer: T5Tokenizer,
    languages: Languages,
    top_k: int = 3,
    **tokenizer_kwargs
) -> list[list[LanguageInfo]]:
    encoding = prepare_inputs_for_model(text, tokenizer, **tokenizer_kwargs)
    outputs = model(encoding["input_ids"]).detach().topk(top_k)

    probabilities = outputs.values.squeeze().tolist()
    language_ids = outputs.indices.squeeze().tolist()

    languages = [[languages[_id] for _id in id_list] for id_list in language_ids]

    for idx, predicted_languages in enumerate(languages):
        for lang_idx, info in enumerate(predicted_languages):
            info["probability"] = probabilities[idx][lang_idx]
    
    return languages
