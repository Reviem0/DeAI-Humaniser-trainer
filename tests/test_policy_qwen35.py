"""Qwen3.5 bases (hybrid linear/full attention, saved as a vision-language model) on a tiny random model:
text-only loading, left-padded batches, LoRA coverage. CPU only; the linear-attention layers use
transformers' PyTorch fallback when flash-linear-attention isn't installed."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from deai import policy

TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def tiny_qwen35_vl():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    cfg = Qwen3_5Config(
        text_config=dict(num_hidden_layers=4, layer_types=TYPES, hidden_size=64, intermediate_size=128,
                         vocab_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                         linear_value_head_dim=16, eos_token_id=1, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64))
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(cfg).float().eval()


def lora_modules(model) -> set[str]:
    peft = get_peft_model(model, LoraConfig(r=4, target_modules=policy.LORA_TARGETS, task_type="CAUSAL_LM"))
    return {n.removeprefix("base_model.model.") for n, m in peft.named_modules() if hasattr(m, "lora_A")}


def test_vl_checkpoint_loads_as_text_only_causal_lm(tmp_path):
    vl = tiny_qwen35_vl()
    vl.save_pretrained(tmp_path)
    lm = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()  # as load_policy does
    assert type(lm).__name__ == "Qwen3_5ForCausalLM"
    ids = torch.randint(2, 512, (1, 12))
    with torch.no_grad():
        assert torch.equal(vl(input_ids=ids).logits, lm(input_ids=ids).logits)


def test_left_padding_leaves_linear_attention_outputs_unchanged(tmp_path):
    # generate() left-pads batches; padding must not leak into the recurrent state.
    tiny_qwen35_vl().save_pretrained(tmp_path)
    lm = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    short, long = torch.randint(2, 512, (1, 7)), torch.randint(2, 512, (1, 12))
    batch = torch.cat([torch.cat([torch.zeros((1, 5), dtype=torch.long), short], 1), long])
    mask = (torch.arange(12) >= 5).long().unsqueeze(0).repeat(2, 1)
    mask[1] = 1
    with torch.no_grad():
        alone = lm(input_ids=short).logits[0, -1]
        padded = lm(input_ids=batch, attention_mask=mask).logits[0, -1]
    assert torch.allclose(alone, padded, atol=1e-4)


def test_lora_reaches_linear_attention_layers(tmp_path):
    tiny_qwen35_vl().save_pretrained(tmp_path)
    mods = lora_modules(AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32))
    for layer, kind in enumerate(TYPES):
        names = {n.rsplit(".", 1)[-1] for n in mods if f".layers.{layer}." in n}
        attn = {"q_proj", "k_proj", "v_proj", "o_proj"} if kind == "full_attention" else \
            {"in_proj_qkv", "in_proj_z", "out_proj"}
        assert names == attn | {"gate_proj", "up_proj", "down_proj"}, (layer, names)


def test_lora_targets_on_qwen3_are_the_original_seven():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                                         vocab_size=512, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=16))
    assert {n.rsplit(".", 1)[-1] for n in lora_modules(model)} == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
