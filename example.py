import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    llm = LLM(path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens) # 生成的最大token数量
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, # 在文本层面应用chat_template
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params) # outputs是一个列表，每个元素是一个字典，包含text和token_ids

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}") # 只打印text，忽略token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano vllm")
    parser.add_argument("--model_path", type=str, 
    default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--enforce-eager", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)
