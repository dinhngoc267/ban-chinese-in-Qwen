import torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import LogitsProcessor

class BanChineseProcessor(LogitsProcessor):
    """ A custom logits processor to ban Chinese characters. """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask = None 
    def __call__(self, input_ids, scores):
        """ Modify the logits before softmax to ban Chinese characters. """
        if self.mask is None:
            vocab_size = scores.shape[-1]
            token_ids = torch.arange(vocab_size)
            decoded_tokens = self.tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)

            self.mask = torch.tensor([
                any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF or 0xF900 <= ord(c) <= 0xFAFF for c in token)
                for token in decoded_tokens
            ], dtype=torch.bool, device=scores.device)

        scores[:, self.mask] = -float("inf")
        return scores


class Generator:
    def __init__(self, max_seq_length=8192):
        self.model = LLM(
            model="Qwen/Qwen2.5-14B-Instruct",  
            tensor_parallel_size=1,
            dtype="auto"
        )
        self.tokenizer = get_tokenizer("Qwen/Qwen2.5-14B-Instruct")
        self.logits_processor = BanChineseProcessor(self.tokenizer)

    def generate(self, query: str):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": query}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.1,
            logits_processors=[self.logits_processor]  
        )

        outputs = self.model.generate([text], sampling_params)
        response = outputs[0].outputs[0].text 

        return response
