from typing import Any, Optional
from numpy.typing import NDArray
import numpy as np
import torch
from style_bert_vits2.constants import Languages
import tritonclient.http as httpclient

BERT_MODEL = {
    Languages.JP: "bert_model_jp",
    Languages.EN: "bert_model_en",
    Languages.ZH: "bert_model_zh",
}

TOKEN_MODEL = {
    Languages.JP: "token_model_jp",
    Languages.EN: "token_model_en",
    Languages.ZH: "token_model_zh",
}

def create_input(name: str, data: Any, datatype: str):
    if data is None:
        return None

    if isinstance(data, str):
        # TEXT hoặc ASSIST_TEXT sẽ encode sang bytes
        data = np.array([data.encode('utf-8')], dtype=np.object_)
    elif isinstance(data, float):
        data = np.array([data], dtype=np.float32)
    elif isinstance(data, int):
        data = np.array([data], dtype=np.int64)
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance( data, np.ndarray):
        data = data
    else:
        raise ValueError(f"Unsupported input type for {name}: {type(data)}")

    inp = httpclient.InferInput(name, data.shape, datatype)
    inp.set_data_from_numpy(data)
    return inp

def call_token_triton( text: str, language: Languages, return_type: Optional[str] = None, tokenize: Optional[str] = None, client: Optional[Any] = None):
    is_close = False
    if client is None:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        is_close = True
    inputs_token = []
    text_input = create_input("TEXT", text, "BYTES")
    if text_input: inputs_token.append(text_input)
    return_input = create_input("RETURN_TENSORS", return_type, "BYTES")
    if return_input: inputs_token.append(return_input)
    tokenize_input = create_input("TOKENIZE", tokenize, "BYTES")
    if tokenize_input: inputs_token.append(tokenize_input)
    result = client.infer(
        model_name=TOKEN_MODEL[language],
        inputs=inputs_token,
    )
    input_ids = result.as_numpy("INPUT_IDS").tolist()
    token_type_ids = result.as_numpy("TOKEN_TYPE_IDS").tolist()
    attention_mask = result.as_numpy("ATTENTION_MASK").tolist()
    tokens = [token.decode('utf-8') for token in result.as_numpy("TOKENS").tolist()]
    if is_close:
        client.close()
    return input_ids, token_type_ids, attention_mask, tokens

def call_bert_triton( input_ids: Any, token_type_ids: Any, attention_mask: Any, language: Languages, client: Optional[Any] = None):
    is_close = False
    if client is None:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        is_close = True
    inputs_token = [
        create_input("INPUT_IDS", input_ids, "INT64"),
        create_input("TOKEN_TYPE_IDS", token_type_ids, "INT64"),
        create_input("ATTENTION_MASK", attention_mask, "INT64")
    ]
    result = client.infer(
        model_name=BERT_MODEL[language],
        inputs=inputs_token,
    )
    if is_close:
        client.close()
    return result.as_numpy("OUTPUT0")

def call_tts_triton( 
        model_name: str, 
        x: NDArray[Any],
        x_lengths: NDArray[Any],
        sid: int,
        tone: NDArray[Any],
        language: NDArray[Any],
        bert: NDArray[Any],
        ja_bert: NDArray[Any],
        en_bert: NDArray[Any],
        style_vec: NDArray[Any],
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: Optional[int] = None,
        sdp_ratio: float = 0.0,
        y: Optional[NDArray[Any]] = None,
        is_jp_extra: bool = True,
        client: Optional[Any] = None,
    ):
    is_close = False
    if client is None:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        is_close = True
    inputs_tts = [
        create_input("X", x, "INT64"),
        create_input("X_LENGTHS", x_lengths, "INT64"),
        create_input("SID", sid, "INT64"),
        create_input("TONE", tone, "INT64"),
        create_input("LANGUAGE", language, "INT64"),
        create_input("STYLE_VEC", style_vec, "FP32"),
        create_input("NOISE_SCALE", noise_scale, "FP32"),
        create_input("LENGTH_SCALE", length_scale, "FP32"),
        create_input("NOISE_SCALE_W", noise_scale_w, "FP32"),
        create_input("MAX_LEN", max_len, "INT64"),
        create_input("SDP_RATIO", sdp_ratio, "FP32"),
        create_input("Y", y, "INT64"),
    ]
    if is_jp_extra:
        inputs_tts.append(create_input("BERT", ja_bert, "FP32"))
    else:
        inputs_tts.append(create_input("BERT", bert, "FP32"))
        inputs_tts.append(create_input("JA_BERT", ja_bert, "FP32"))
        inputs_tts.append(create_input("EN_BERT", en_bert, "FP32"))
    inputs_tts = filter( lambda x: x is not None, inputs_tts)
    result = client.infer(
        model_name=model_name,
        inputs=list(inputs_tts),
    )
    if is_close:
        client.close()
    return result.as_numpy("OUTPUT0")