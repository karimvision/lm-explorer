import itertools

from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHeadModel
import torch
import tqdm

from lm_explorer.lm.language_model import LanguageModel
from lm_explorer.util.cache import LRUCache
from lm_explorer.util.sampling import random_sample

model_345 = "/mnt/ed/lm-explorer/models/pytorch_models/345M/"
model_774 = "/mnt/ed/lm-explorer/models/pytorch_models/774M/"
model_1558 = "/mnt/ed/lm-explorer/models/pytorch_models/1558M/"

class GPT2LanguageModel(LanguageModel):
    def __init__(self, cache_size: int = 0, model_name: str = '117M',device='cuda') -> None:
        """
        Each cache element is about 8MB, so size accordingly.
        """
        # Cache stores tuples, so default value is a tuple
        self._cache = LRUCache(cache_size, default_value=(None, None))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if model_name == '345M':
            self.model = GPT2LMHeadModel.from_pretrained(model_345)
        elif model_name == '774M':
            self.model = GPT2LMHeadModel.from_pretrained(model_774)
        else:
            exit("model name not found")
        self.model = self.model.to(device)
        self.device = device
        # The end of text marker.
        self.END_OF_TEXT = self.tokenizer.encoder["<|endoftext|>"]

    def predict(self, previous: str, next: str = None) -> torch.Tensor:

        past_logits, past = self._cache[previous]

        # CASE 1: Previously seen input, no next
        if next is None and past is not None:
            return past_logits

        # CASE 2: Previously seen input, yes next
        elif past is not None:
            token_ids = self.tokenizer.encode(next)
        # CASE 3: Brand new input, no next
        elif next is None:
            token_ids = self.tokenizer.encode(previous)
        # CASE 4: Brand new input, yes next
        else:
            token_ids = self.tokenizer.encode(previous) + self.tokenizer.encode(next)

        inputs = torch.LongTensor([token_ids]).to(self.device)

        logits, present = self.model(inputs, past=past)
        logits = logits[0, -1]

        key = previous if next is None else previous + next
        self._cache[key] = logits, present

        return logits
    
    def get_sentence_loss(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        # add end_of_text
        tensor_input = torch.tensor([ [50256]  +  self.tokenizer.convert_tokens_to_ids(tokenize_input)]).to(self.device)
        loss=self.model(tensor_input, lm_labels=tensor_input)
        return loss.item() * len(tokenize_input)

    def __getitem__(self, index: int) -> str:
        return self.tokenizer.decode([index])

    def generate(self, seed: str = "", max_len: int = None) -> str:

        output = seed
        logits = self.predict(seed)

        if max_len is None:
            it = tqdm.tqdm(itertools.count())
        else:
            it = tqdm.trange(max_len)

        for _ in it:
            next_id = random_sample(logits)
            next_word = self[next_id]

            print(next_word)

            if next_word == "<|endoftext|>":
                break

            logits = self.predict(output, next_word)
            output += next_word

        return output

#         # Sample from logits
#         d = torch.distributions.Categorical(logits=logits[0, -1])
#         next_id = d.sample().item()

#         if next_id == END_OF_TEXT:
#             break

#         token_ids.append(next_id)
#         inputs = torch.LongTensor([[next_id]])

#     # Decode
#     return tokenizer.decode(token_ids)

# print(generate(seed=SEED, num_steps=50))
