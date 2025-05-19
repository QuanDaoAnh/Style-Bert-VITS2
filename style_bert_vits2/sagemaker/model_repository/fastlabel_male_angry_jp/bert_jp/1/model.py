import json

from transformers import (
    AutoModelForMaskedLM,
)

try:
    import torch
except ModuleNotFoundError as error:
    raise RuntimeError("Missing/Incomplete PyTorch package installation") from error

import triton_python_backend_utils as pb_utils
from typing import Any, Optional
from numpy.typing import NDArray

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.models import commons
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
from style_bert_vits2.nlp import (
    cleaned_text_to_sequence,
    clean_text,
)
import numpy as np

class TritonPythonModel:

    def initialize(self, args):
        self._model_name = args["model_name"]
        for_model = "for '" + self._model_name + "'"
        self._logger = pb_utils.Logger
        self._logger.log_info("Initializing model instance " + for_model)

        self._device_id = args["model_instance_device_id"]
        if self._device_id is None or self._device_id == -1:
            # CPU instance
            self._device = "cpu"
        else:
            # GPU instance
            self._device = "cuda"
        self._version = args["model_version"]
        self._model_config = json.loads(args["model_config"])
        self._model_repository = args["model_repository"]

        self._logger.log_info("Model path " + str(self._model_repository))
        self._logger.log_info("Model device " + str(self._device_id))

        self._model_path = self._model_config["parameters"]["model_file"]["string_value"]

        bert_models.load_model(
            language=Languages.JP,
            pretrained_model_name_or_path=self._model_path,
        )
        bert_models.load_tokenizer(
            language=Languages.JP,
            pretrained_model_name_or_path=self._model_path,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            version_input = pb_utils.get_input_tensor_by_name(request, "VERSION")
            version = bytes(version_input.as_numpy().tolist()[0]).decode('utf-8')

            text_input = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text = bytes(text_input.as_numpy().tolist()[0]).decode('utf-8')

            assist_text_input = pb_utils.get_input_tensor_by_name(request, "ASSIST_TEXT")
            if assist_text_input is not None:
                assist_text = bytes(assist_text_input.as_numpy().tolist()[0]).decode('utf-8')
            else:
                assist_text = None
            assist_text_weight_input = pb_utils.get_input_tensor_by_name(request, "ASSIST_TEXT_WEIGHT")
            assist_text_weight = assist_text_weight_input.as_numpy().tolist()[0]

            given_phone_input = pb_utils.get_input_tensor_by_name(request, "GIVEN_PHONE")
            if given_phone_input is not None:
                given_phone = [bytes(text.tolist()).decode('utf-8') for text in given_phone_input.as_numpy()]
            else:
                given_phone = None

            given_tone_input = pb_utils.get_input_tensor_by_name(request, "GIVEN_TONE")
            if given_tone_input is not None:
                given_tone = given_tone_input.as_numpy().tolist()
            else:
                given_tone = None

            add_blank_input = pb_utils.get_input_tensor_by_name(request, "ADD_BLANK")
            add_blank = add_blank_input.as_numpy().tolist()[0] == 1

            skip_start_input = pb_utils.get_input_tensor_by_name(request, "SKIP_START")
            skip_start = skip_start_input.as_numpy().tolist()[0] == 1

            skip_end_input = pb_utils.get_input_tensor_by_name(request, "SKIP_END")
            skip_end = skip_end_input.as_numpy().tolist()[0] == 1

            is_jp_extra = version.endswith("JP-Extra")
            bert, ja_bert, en_bert, phones, tones, lang_ids = self.get_text(
                text,
                use_jp_extra=is_jp_extra,
                add_blank=add_blank,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                given_phone=given_phone,
                given_tone=given_tone,
            )
            if skip_start:
                phones = phones[3:]
                tones = tones[3:]
                lang_ids = lang_ids[3:]
                bert = bert[:, 3:]
                ja_bert = ja_bert[:, 3:]
                en_bert = en_bert[:, 3:]
            if skip_end:
                phones = phones[:-2]
                tones = tones[:-2]
                lang_ids = lang_ids[:-2]
                bert = bert[:, :-2]
                ja_bert = ja_bert[:, :-2]
                en_bert = en_bert[:, :-2]
            
            output_bert = pb_utils.Tensor("BERT", np.array([bert], np.float32))
            output_ja_bert = pb_utils.Tensor("JA_BERT", np.array([ja_bert], np.float32))
            output_en_bert = pb_utils.Tensor("EN_BERT", np.array([en_bert], np.float32))
            output_phones = pb_utils.Tensor("PHONES", np.array([phones]))
            output_tones = pb_utils.Tensor("TONE", np.array([tones]))
            output_lang_ids = pb_utils.Tensor("LANG_IDS", np.array([lang_ids]))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_bert, output_ja_bert, output_en_bert, output_phones, output_tones, output_lang_ids]
            )
            responses.append(inference_response)
        return responses

    def get_text(
        self, 
        text: str,
        use_jp_extra: bool, 
        add_blank: bool,
        assist_text: Optional[str] = None,
        assist_text_weight: float = 0.7,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
    ) -> tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], list[Any], list[Any], list[Any]
    ]:
        norm_text, phone, tone, word2ph = self.clean_text_with_given_phone_tone(
            text,
            Languages.JP,
            given_phone=given_phone,
            given_tone=given_tone,
            use_jp_extra=use_jp_extra,
            raise_yomi_error=False,
        )
        phone, tone, language = cleaned_text_to_sequence(phone, tone, Languages.JP)

        if add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_ori = self.extract_bert_feature(
            norm_text,
            word2ph,
            Languages.JP,
            device=self._device, 
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
        )
        del word2ph
        assert bert_ori.shape[-1] == len(phone), phone

        bert = np.zeros((1024, len(phone)))
        ja_bert = bert_ori.detach().cpu().numpy()
        en_bert = np.zeros((1024, len(phone)))

        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
        return bert, ja_bert, en_bert, phone, tone, language

    def clean_text_with_given_phone_tone(
        self, 
        text: str,
        language: Languages,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        use_jp_extra: bool = True,
        raise_yomi_error: bool = False,
    ) -> tuple[str, list[str], list[int], list[int]]:
        """
        テキストをクリーニングし、音素に変換する
        変換時、given_phone や given_tone が与えられた場合はそれを調整して使う

        Args:
            text (str): クリーニングするテキスト
            language (Languages): テキストの言語
            given_phone (Optional[list[int]], optional): 読み上げテキストの読みを表す音素列。指定する場合は given_tone も別途指定が必要. Defaults to None.
            given_tone (Optional[list[int]], optional): アクセントのトーンのリスト. Defaults to None.
            use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
            raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

        Returns:
            tuple[str, list[str], list[int], list[int]]: クリーニングされたテキストと、音素・アクセント・元のテキストの各文字に音素が何個割り当てられるかのリスト
        """

        # 与えられたテキストをクリーニング
        norm_text, phone, tone, word2ph = clean_text(
            text,
            language,
            use_jp_extra=use_jp_extra,
            raise_yomi_error=raise_yomi_error,
        )

        # phone と tone の両方が与えられた場合はそれを使う
        if given_phone is not None and given_tone is not None:
            # 指定された phone と指定された tone 両方の長さが一致していなければならない
            if len(given_phone) != len(given_tone):
                raise InvalidPhoneError(
                    f"Length of given_phone ({len(given_phone)}) != length of given_tone ({len(given_tone)})"
                )
            # 与えられた音素数と pyopenjtalk で生成した読みの音素数が一致しない
            if len(given_phone) != sum(word2ph):
                # 日本語の場合、len(given_phone) と sum(word2ph) が一致するように word2ph を適切に調整する
                # 他の言語は word2ph の調整方法が思いつかないのでエラー
                if language == Languages.JP:
                    from style_bert_vits2.nlp.japanese.g2p import adjust_word2ph

                    # use_jp_extra でない場合は given_phone 内の「N」を「n」に変換
                    if not use_jp_extra:
                        given_phone = [p if p != "N" else "n" for p in given_phone]
                    # clean_text() から取得した word2ph を調整結果で上書き
                    word2ph = adjust_word2ph(word2ph, phone, given_phone)
                    # 上記処理により word2ph の合計が given_phone の長さと一致するはず
                    # それでも一致しないとしたら、len(generated_phone) に比べて len(given_phone) があまりに少なすぎて、
                    # 各文字ごとに最低 1 以上の音素を割り当てることが不可能だったことを意味する
                    # 通常無理やりにでも辻褄を合わせるため発生しないはずだが、どうしても一致しない場合はエラーとする
                    if len(given_phone) != sum(word2ph):
                        raise InvalidPhoneError(
                            f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                        )
                else:
                    raise InvalidPhoneError(
                        f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                    )
            phone = given_phone
            # 生成あるいは指定された phone と指定された tone 両方の長さが一致していなければならない
            if len(phone) != len(given_tone):
                raise InvalidToneError(
                    f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
                )
            tone = given_tone

        # tone だけが与えられた場合は clean_text() で生成した phone と合わせて使う
        elif given_tone is not None:
            # 生成した phone と指定された tone 両方の長さが一致していなければならない
            if len(phone) != len(given_tone):
                raise InvalidToneError(
                    f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
                )
            tone = given_tone

        # 日本語のみ、g2p 処理では対応しているが現行モデルでは対応していない特定音素を変換 (フォールバック)
        if language == Languages.JP:

            # 音素変換マップ
            PHONE_CONVERSION_MAP = {
                "kw": ("k", "u", "w"),  # 「クヮ」→「クワ」
                "gw": ("g", "u", "w"),  # 「グヮ」→「グワ」
                "fy": ("hy",),  # 「フュ」→「ヒュ」
            }

            # 変換が必要な音素のインデックスを収集
            conversion_indices: list[tuple[int, str]] = []
            for i, p in enumerate(phone):
                if p in PHONE_CONVERSION_MAP:
                    conversion_indices.append((i, p))

            # 音素変換が必要な場合のみ処理を実行
            if conversion_indices:

                # インデックスは後ろから処理することで、
                # 前の変換による位置ずれの影響を受けないようにする
                for orig_idx, orig_phone in reversed(conversion_indices):

                    # 変換後の音素を取得
                    converted_phones = PHONE_CONVERSION_MAP[orig_phone]

                    # phone リストの更新
                    ## スライスで置換すると要素数が変化する
                    phone[orig_idx : orig_idx + 1] = list(converted_phones)

                    # tone リストの更新
                    ## 元の音素のトーンを、変換後の音素全てに適用
                    orig_tone = tone[orig_idx]
                    tone[orig_idx : orig_idx + 1] = [orig_tone] * len(converted_phones)

                    # word2ph リストの更新
                    ## 元の音素が属していた文字のインデックスを特定
                    char_idx = 0
                    phone_count = 0
                    for i, count in enumerate(word2ph):
                        if phone_count + count > orig_idx:
                            char_idx = i
                            break
                        phone_count += count

                    # 該当する文字の音素数を更新
                    ## 1つの音素が3つの音素に変換されるので、2つ増える
                    word2ph[char_idx] += len(converted_phones) - 1

            # ここでは必ず音素数が一致するはず
            assert len(phone) == len(tone) == sum(word2ph)

        return norm_text, phone, tone, word2ph

    def extract_bert_feature(
        self,
        text: str,
        word2ph: list[int],
        language: Languages,
        device: str,
        assist_text: Optional[str] = None,
        assist_text_weight: float = 0.7,
    ) -> torch.Tensor:
        """
        テキストから BERT の特徴量を抽出する (PyTorch 推論)

        Args:
            text (str): テキスト
            word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
            language (Languages): テキストの言語
            device (str): 推論に利用するデバイス
            assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
            assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

        Returns:
            torch.Tensor: BERT の特徴量
        """

        if language == Languages.JP:
            text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
            if assist_text:
                assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = bert_models.load_model(language).to(device)

        style_res_mean = None
        with torch.no_grad():
            tokenizer = bert_models.load_tokenizer(language)
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)  # type: ignore
            res = model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0]
            if assist_text:
                style_inputs = tokenizer(assist_text, return_tensors="pt")
                for i in style_inputs:
                    style_inputs[i] = style_inputs[i].to(device)  # type: ignore
                style_res = model(**style_inputs, output_hidden_states=True)
                style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0]
                style_res_mean = style_res.mean(0)

        assert len(word2ph) == len(text) + 2, text
        word2phone = torch.LongTensor(word2ph).to(device)
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = res * (1 - assist_text_weight) + style_res_mean * assist_text_weight
        else:
            repeat_feature = res
        phone_level_feature = torch.repeat_interleave(repeat_feature, word2phone, 0)
        return phone_level_feature.T

    def finalize(self):
        self._logger.log_info("Removing model instance for '" + self._model_name + "'")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class InvalidPhoneError(ValueError):
    pass

class InvalidToneError(ValueError):
    pass