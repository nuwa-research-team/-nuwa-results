"""Greedy signed-length decoding. Only the model chooses signs and magnitudes."""
import re
import torch
from eval_packing import parse_p1


class ArrayGrammar:
    def __init__(self, tokenizer, count):
        if count < 1:
            raise ValueError("Expected a nonempty query")
        self.count = count
        self.eos = tokenizer.eos_token_id
        self.ids = {}
        for char in "0123456789-[],":
            token, = tokenizer.encode(char, add_special_tokens=False)
            assert tokenizer.decode([token]) == char
            self.ids[char] = token
        for punctuation in ("[-", ",-"):
            pieces = tokenizer.encode(punctuation, add_special_tokens=False)
            if len(pieces) == 1 and tokenizer.decode(pieces) == punctuation:
                self.ids[punctuation] = pieces[0]
        self.text = {v:k for k,v in self.ids.items()}

    def allowed(self, batch_id, input_ids):
        prefix = "".join(self.text[t] for t in input_ids.tolist())
        if not prefix:
            choices = ["[", "[-"]
        elif prefix.endswith("]"):
            return [self.eos]
        else:
            current = re.split(r"[\[,]", prefix)[-1]
            if current in ("", "-"):
                choices = list("123456789")
                if not current:
                    choices.append("-")
            else:
                value = abs(int(current))
                choices = [str(i) for i in range(10) if value * 10 + i <= 65535]
                if value >= 40:
                    choices.extend(["]"] if prefix.count(",")+1 == self.count else [",", ",-"])
        return [self.ids[c] for c in choices if c in self.ids]


@torch.inference_mode()
def generate(model, tokenizer, pack, count):
    grammar = ArrayGrammar(tokenizer, count)
    prefix = pack["inputs_embeds"]
    budget = 7 * count + 2  # brackets + EOS, 5 digits, optional sign, separators
    if prefix.shape[1] + budget > model.config.max_position_embeddings:
        raise ValueError("Context budget too small")
    ids = model.generate(
        input_ids=torch.empty((1, 0), dtype=torch.long, device=prefix.device),
        inputs_embeds=prefix, attention_mask=pack["attention_mask"],
        prefix_allowed_tokens_fn=grammar.allowed, do_sample=False, num_beams=1,
        max_new_tokens=budget, eos_token_id=grammar.eos, pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )[0].tolist()
    if ids[-1] != grammar.eos:
        raise ValueError("No terminal EOS")
    raw = tokenizer.decode(ids[:-1], clean_up_tokenization_spaces=False)
    parsed = parse_p1(raw, count)
    if parsed["status"] != "valid":
        raise ValueError((raw, count))
    return raw, parsed["parsed_values"]
