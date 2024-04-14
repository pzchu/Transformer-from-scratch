# Copyright (c) 2022 Zhipu.AI
import copy
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor

import torch

from modelscope.models.base import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks


from transformers import AutoTokenizer, AutoModel
from modeling_chatglm import InvalidScoreLogitsProcessor


@MODELS.register_module(Tasks.chat, module_name='chatglm26b')
class ChatGLM26bForTextGeneration(TorchModel):

    def __init__(self, model_dir: str, device_map: None, *args, **kwargs):
        """initialize the chatglm6b from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.logger = get_logger()
        # loading tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # loading model
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, device_map=device_map).half().cuda()


    def forward(self, input: Dict) -> Dict:
        response, history = self.chat(input)
        return {OutputKeys.RESPONSE:response, OutputKeys.HISTORY: history}

    def chat(self, input: Dict) -> Dict:
        text = input['text']
        history = input['history']
        # args
        if 'max_length' in input:
            max_length = input['max_length']
        else:
            max_length = 2048

        if 'top_p' in input:
            top_p = input['top_p']
        else:
            top_p = 0.7
        
        if 'temperature' in input:
            temperature = input['temperature']
        else:
            temperature = 0.95

        if 'num_beams' in input:
            num_beams = input['num_beams']
        else:
            num_beams = 1
        
        if 'do_sample' in input:
            do_sample = input['do_sample']
        else:
            do_sample = True

        if type(history) == torch.Tensor:
            history = history.tolist()
        response, history = self.model.chat(self.tokenizer, text, history, max_length=max_length, temperature=temperature, num_beams=num_beams, do_sample=do_sample)
        self.logger.info('Generation finished.')
        return response, history

    def quantize(self, bits: int):
        self.model = self.model.quantize(bits)
        return self


@PIPELINES.register_module(
    group_key=Tasks.chat,
    module_name='chatglm26b-text-generation')
class ChatGLM26bTextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: [Preprocessor] = None,
                 *args,
                 **kwargs):
        device_map = kwargs.pop('device_map', None)
        model = ChatGLM26bForTextGeneration(model, device_map=device_map) if isinstance(model,
                                                               str) else model
        self.model = model
        self.model.eval()

        super().__init__(model=model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        return self.model(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input