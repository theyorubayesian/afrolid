import requests
import tarfile
import tempfile
from pathlib import Path
from platformdirs import user_cache_dir
from typing import Optional

import sentencepiece as spm
import torch

from .conversion import create_pytorch_state_dict
from .language_info import LanguageInfo
from .model import AfroLIDModel

AFROLID_CACHE_DIR = Path(user_cache_dir('afrolid'))
AFROLID_CACHE_DIR.mkdir(parents=True, exist_ok=True)

AFROLID_DOWNLOAD_URL = 'https://demos.dlnlp.ai/afrolid/afrolid_model.tar.gz'


def download_and_extract_model(path: Optional[str] = None) -> None:
    if any(
        (Path(p) / "afrolid_model/afrolid_v1_checkpoint.pt").exists()
        for p in [path, AFROLID_CACHE_DIR] if p is not None
    ):
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        download_path = Path(temp_dir) / 'afrolid_model.tar.gz'

        response = requests.get(AFROLID_DOWNLOAD_URL, stream=True)
        with download_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(AFROLID_CACHE_DIR if path is None else path)


def load_afrolid_artifacts(download_path: Optional[str] = None) -> tuple[AfroLIDModel, spm.SentencePieceProcessor, LanguageInfo]:
    afrolid = AfroLIDModel()

    model_path = Path(download_path) if download_path else AFROLID_CACHE_DIR

    if (model_path / "torch_model.bin").exists():
        afrolid.load_state_dict(torch.load(model_path / "torch_model.bin"), strict=False)
    else:
        download_and_extract_model(download_path)
        fairseq_dict = torch.load(model_path / "afrolid_model/afrolid_v1_checkpoint.pt")
        conversion_result = create_pytorch_state_dict(fairseq_dict)

        # TODO: @theyorubayesian - Sanity checks
        torch_dict = conversion_result["new_state_dict"]
        torch.dump(torch_dict, model_path / "torch_model.bin")

        afrolid.load_state_dict(torch_dict, strict=False)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path / "afrolid_model/afrolid_spm_517_bpe.model"))

    language_info = LanguageInfo(model_path / "afrolid_model/dict_label.txt")

    return afrolid, tokenizer, language_info
    

def prepare_input_for_model(text: str, tokenizer: spm.SentencePieceProcessor) -> torch.Tensor:
    return torch.IntTensor([x+1 for x in tokenizer.EncodeAsIds(text)] + [2])
