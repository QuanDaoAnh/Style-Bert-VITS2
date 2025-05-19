import importlib
import json
import os
import numpy as np
from style_bert_vits2.models.infer import get_net_g
from style_bert_vits2.models.hyper_parameters import HyperParameters
from torch import from_dlpack
from numpy.typing import NDArray
from typing import Any
import torch

try:
    import torch
except ModuleNotFoundError as error:
    raise RuntimeError("Missing/Incomplete PyTorch package installation") from error

import triton_python_backend_utils as pb_utils

DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0

class TritonPythonModel:

    def initialize(self, args):
        self._model_name = args["model_name"]
        for_model = "for '" + self._model_name + "'"
        self._logger = pb_utils.Logger
        self._logger.log_info("Initializing model instance " + for_model)

        self._device_id = args["model_instance_device_id"]
        if self._device_id is None or self._device_id == -1:
            self._device = "cpu"
        else:
            self._device = f"cuda:{self._device_id}"
        self._version = args["model_version"]
        self._model_config = json.loads(args["model_config"])
        self._model_repository = args["model_repository"]

        self._logger.log_info("Model path " + str(self._model_repository))
        self._logger.log_info("Model device " + str(self._device_id))

        # Config and HyperParameters
        self._config = os.path.join( self._model_repository, self._model_config["parameters"]["config_path"]["string_value"])
        self._hps = HyperParameters.load_from_json(self._config)

        # style vector
        self.style_vec_path = os.path.join( self._model_repository, self._model_config["parameters"]["style_vec_path"]["string_value"])
        self.style_vectors: NDArray[Any] = np.load(self.style_vec_path)

        # style2id
        num_styles: int = self._hps.data.num_styles
        if hasattr(self._hps.data, "style2id"):
            self.style2id: dict[str, int] = self._hps.data.style2id
        else:
            self.style2id: dict[str, int] = {str(i): i for i in range(num_styles)}
        if len(self.style2id) != num_styles:
            raise ValueError(
                f"Number of styles ({num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        # Model
        self._model_path = os.path.join(self._model_repository, self._version, self._model_config["parameters"]["model_file"]["string_value"])
        self._model = get_net_g(
            model_path=self._model_path,
            version=self._hps.version,
            device=self._device,
            hps=self._hps,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get version
            version_input = pb_utils.get_input_tensor_by_name(request, "VERSION")
            version = bytes(version_input.as_numpy().tolist()[0]).decode('utf-8')
            is_jp_extra = version.endswith("JP-Extra")

            # Get input
            x = from_dlpack(pb_utils.get_input_tensor_by_name(request, "X").to_dlpack()).to(self._device)
            x_lengths = torch.LongTensor([x.size(1)]).to(self._device)
            sid = from_dlpack(pb_utils.get_input_tensor_by_name(request, "SID").to_dlpack()).to(self._device)
            tone = from_dlpack(pb_utils.get_input_tensor_by_name(request, "TONE").to_dlpack()).to(self._device)
            language = from_dlpack(pb_utils.get_input_tensor_by_name(request, "LANGUAGE").to_dlpack()).to(self._device)
            bert = from_dlpack(pb_utils.get_input_tensor_by_name(request, "BERT").to_dlpack()).to(self._device)
            ja_bert = from_dlpack(pb_utils.get_input_tensor_by_name(request, "JA_BERT").to_dlpack()).to(self._device)
            en_bert = from_dlpack(pb_utils.get_input_tensor_by_name(request, "EN_BERT").to_dlpack()).to(self._device)

            # Get Style Vector
            style_vec_input = pb_utils.get_input_tensor_by_name(request, "STYLE_VEC")
            if style_vec_input is not None:
                style_vec = style_vec_input.as_numpy()
                style_weight_input = pb_utils.get_input_tensor_by_name(request, "STYLE_WEIGHT")
                if style_weight_input is not None:
                    style_weight = style_weight_input.as_numpy()[0]
                else:
                    style_weight = DEFAULT_STYLE_WEIGHT
                mean = self.style_vectors[0]
                style_vec = mean + (style_vec - mean) * style_weight
            else:
                style_input = pb_utils.get_input_tensor_by_name(request, "STYLE")
                if style_input is not None:
                    style = bytes(style_input.as_numpy().tolist()[0]).decode('utf-8')
                else:
                    style = DEFAULT_STYLE
                style_weight_input = pb_utils.get_input_tensor_by_name(request, "STYLE_WEIGHT")
                if style_weight_input is not None:
                    style_weight = style_weight_input.as_numpy()[0]
                else:
                    style_weight = DEFAULT_STYLE_WEIGHT

                styles_input = pb_utils.get_input_tensor_by_name(request, "STYLES")
                if styles_input is not None:
                    styles = [bytes(text.tolist()).decode('utf-8') for text in styles_input.as_numpy()]
                else:
                    styles = []

                style_weights_input = pb_utils.get_input_tensor_by_name(request, "STYLE_WEIGHTS")
                if style_weights_input is not None:
                    style_weights = style_weights_input.as_numpy().tolist()
                else:
                    style_weights = []
                style_vec = self.get_style_vector( style, style_weight, styles, style_weights)
            style_vec = torch.from_numpy(style_vec).to(self._device).unsqueeze(0)

            # Get Another Param
            noise_scale = from_dlpack(pb_utils.get_input_tensor_by_name(request, "NOISE_SCALE").to_dlpack()).item()
            length_scale = from_dlpack(pb_utils.get_input_tensor_by_name(request, "LENGTH_SCALE").to_dlpack()).item()
            noise_scale_w = from_dlpack(pb_utils.get_input_tensor_by_name(request, "NOISE_SCALE_W").to_dlpack()).item()
            sdp_ratio = from_dlpack(pb_utils.get_input_tensor_by_name(request, "SDP_RATIO").to_dlpack()).item()
            max_len_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_LEN")
            if max_len_tensor is not None:
                max_len = from_dlpack(max_len_tensor.to_dlpack()).item()
            else:
                max_len = None
            y_tensor = pb_utils.get_input_tensor_by_name(request, "Y")
            if y_tensor is not None:
                y = from_dlpack(y_tensor.to_dlpack()).item()
            else:
                y = None
            if not is_jp_extra:
                output_array = self._model.infer(
                    x, x_lengths, sid, tone, language, bert, ja_bert, en_bert, 
                    style_vec=style_vec,
                    length_scale=length_scale,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    max_len=max_len,
                    y=y
                )
            else:
                output_array = self._model.infer(
                    x, x_lengths, sid, tone, language, ja_bert,
                    style_vec=style_vec,
                    length_scale=length_scale,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    max_len=max_len,
                    y=y
                )
            audio_tensor = pb_utils.Tensor("AUDIO", output_array[0][0, 0].to( torch.float32).data.cpu().numpy())
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[audio_tensor]
            )
            responses.append(inference_response)
        return responses

    def get_style_vector(self, 
                                    style: str = DEFAULT_STYLE,
                                    style_weight: float = DEFAULT_STYLE_WEIGHT,
                                    styles: list[str] = [],
                                    style_weights: list[float] = []
                                ) -> NDArray[Any]:
        if len(styles) != len(style_weights):
            raise ValueError("styles and style_weights must have the same length")
        if len(styles) == 0:
            style_id = self.style2id.get(style)
            style_vector = self.get_style_vector_single(style_id, style_weight)
        else:
            style_ids = [self.style2id.get(style_name) for style_name in styles]
            style_vector = self.get_style_vector_mix(style_ids, style_weights)
        return style_vector

    def get_style_vector_single(self, style_id: int, weight: float = 1.0) -> NDArray[Any]:
        """
        スタイルベクトルを取得する。

        Args:
            style_id (int): スタイル ID (0 から始まるインデックス)
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.

        Returns:
            NDArray[Any]: スタイルベクトル
        """
        mean = self.style_vectors[0]
        style_vec = self.style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def get_style_vector_mix(self, style_ids: list[int], style_weights: list[float]) -> NDArray[Any]:
        """
        スタイルベクトルを取得する。
        Args:
            style_id (int): スタイル ID (0 から始まるインデックス)
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.
        Returns:
            NDArray[Any]: スタイルベクトル
        """
        style_vec = None
        for i, style_id in enumerate(style_ids):
            style_data = self.style_vectors[style_id] * style_weights[i]
            if style_vec is None:
                style_vec = style_data
            else:
                style_vec += style_data
        if style_vec is None:
            raise ValueError("Empty style")
        return style_vec

    def finalize(self):
        self._logger.log_info("Removing model instance for '" + self._model_name + "'")
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
