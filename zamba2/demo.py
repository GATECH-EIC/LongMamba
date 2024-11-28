import torch
import sys
sys.path.append('..')
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from zamba2.modeling_zamba2 import Zamba2ForCausalLM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B")
model = Zamba2ForCausalLM.from_pretrained("Zyphra/Zamba2-1.2B").to(device=device)

merge_config = {
    "model_arch": "vanilla",
    "save_para4debug": True,
}

tensors = model.generate(
    torch.tensor(tokenizer.encode("Hello, how are you?")).unsqueeze(0).to(device),
    do_sample=False,
    max_new_tokens=5,
    top_k=50,
    top_p=0.95,
    temperature=1.,
    merge_config=merge_config,
)

print(tensors)
print(model.params_for_debug)