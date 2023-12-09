import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from argparse import ArgumentParser

device = "cuda"
eos = "<|endoftext|>"
tokenizer = AutoTokenizer.from_pretrained("havenhq/mamba-chat")
tokenizer.eos_token = eos
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta"
).chat_template

model = MambaLMHeadModel.from_pretrained(
    "havenhq/mamba-chat", device="cuda", dtype=torch.float16
)


def chat_with_mamba(user_message, history: list[list[str]]):
    history_dict: list[dict[str, str]] = []
    for user_m, assistant_m in history:
        history_dict.append(dict(role="user", content=user_m))
        history_dict.append(dict(role="assistant", content=assistant_m))
    history_dict.append(dict(role="user", content=user_message))

    input_ids = tokenizer.apply_chat_template(
        history_dict, return_tensors="pt", add_generation_prompt=True
    ).to("cuda")

    out = model.generate(
        input_ids=input_ids,
        max_length=2000,
        temperature=0.9,
        top_p=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.batch_decode(out)
    assistant_message = (
        decoded[0].split("<|assistant|>\n")[-1].replace(eos, "")
    )
    return assistant_message


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="share your instance publicly through gradio",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    demo = gr.ChatInterface(
        fn=chat_with_mamba,
        examples=[
            "Explain what is state space model",
            "Nice to meet you!",
            "'Mamba is way better than ChatGPT.' Is this statement correct?",
        ],
        title="Mamba Chat",
    )
    demo.launch(server_port=args.port, share=args.share)
