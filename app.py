import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default='cuda', help='Device to run the model on')
    parser.add_argument("--model", type=str, default='havenhq/mamba-chat', help='Model to use')
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="share your instance publicly through gradio",
    )
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    return args


if __name__ == "__main__":
    args = get_args()

    device = args.device
    model_name = args.model
    eos = "<|endoftext|>"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.eos_token = eos
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta"
    ).chat_template

    model = MambaLMHeadModel.from_pretrained(
        model_name, device=device, dtype=torch.float16
    )

    def chat_with_mamba(
        user_message,
        history: list[list[str]],
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_length: int = 2000,
    ):
        history_dict: list[dict[str, str]] = []
        for user_m, assistant_m in history:
            history_dict.append(dict(role="user", content=user_m))
            history_dict.append(dict(role="assistant", content=assistant_m))
        history_dict.append(dict(role="user", content=user_message))

        input_ids = tokenizer.apply_chat_template(
            history_dict, return_tensors="pt", add_generation_prompt=True
        ).to(device)

        out = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.batch_decode(out)
        assistant_message = (
            decoded[0].split("<|assistant|>\n")[-1].replace(eos, "")
        )
        return assistant_message
    

    demo = gr.ChatInterface(
        fn=chat_with_mamba,
        # examples=[
        #     "Explain what is state space model",
        #     "Nice to meet you!",
        #     "'Mamba is way better than ChatGPT.' Is this statement correct?",
        # ],
        additional_inputs=[
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="temperature"),
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="top_p"),
            gr.Number(value=2000, label="max_length"),
        ],
        title="Mamba Chat",
    )
    demo.launch(server_port=args.port, share=args.share)
