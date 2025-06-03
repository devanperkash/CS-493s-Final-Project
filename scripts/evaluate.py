import torch
import argparse
from models import StudentModel, TeacherModel
from distill_utils import get_device
from transformers import AutoTokenizer

# For GSM8K evaluation:
import evaluate as hf_evaluate
from datasets import load_dataset

# For HellaSwag via lm_harness:
from lm_eval.base import BaseLM
from lm_eval.api import evaluate as lm_eval_run
from lm_eval.tasks import HellaSwag

class StudentCausalLMWrapper(BaseLM):
    """
    A thin wrapper so that lm_eval_harness can call `model.loglikelihoods()` or `model.generate()`.
    We just implement .generate() and let HellaSwag use that.
    """
    def __init__(self, student_model, tokenizer, device):
        self.model = student_model
        self.tokenizer = tokenizer
        self.device = device

        # we’ll put the student in eval mode
        self.model.eval()

    @property
    def eot_token_id(self):
        # end-of-text token: take tokenizer.eos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # define a sensible generation cap
        return 100  # or however many tokens you want

    @property
    def max_gen_toks(self):
        # for HellaSwag, typically you need to generate a few tokens to score each completion
        return 2

    @property
    def batch_size(self):
        return 1  # harness will batch‐ify if needed

    def _model_call(self, inps):
        """
        This should return, for each input string in `inps`, the raw logits tensor of shape (1, seq_len, V).
        HellaSwag uses log‐likelihood, not free‐generation, but the harness will handle that internally.
        """
        # Tokenize the prompt
        inputs = self.tokenizer(inps, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"]            # (1, seq_len)
        attn_mask = inputs["attention_mask"]        # (1, seq_len)

        # Pass through student model; we need forward() to return logits for all positions
        # but our StudentModel.forward expects (input_ids, attention_mask)
        with torch.no_grad():
            logits = self.model.forward(input_ids, attention_mask=attn_mask)  # (1, seq_len, V)
        return logits

    def generate(self, inps, max_length=None, eos_token_id=None):
        """
        Called by harness for sampling‐based tasks.
        Returns a list of generated-token lists (not strings).
        """
        if max_length is None:
            max_length = self.max_length
        if eos_token_id is None:
            eos_token_id = self.eot_token_id

        # We can simply call our model’s generate_text, which returns strings.
        # But HellaSwag wants token IDs. So:
        results = []
        for prompt in inps:
            # generate_text returns a Python list of string(s). We take the single string, re‐tokenize it to IDs.
            out_str = self.model.generate_text([prompt], max_length=max_length)[0]
            token_ids = self.tokenizer.encode(out_str, add_special_tokens=False)
            results.append(token_ids)
        return results

def evaluate_student(args):
    device = get_device()

    # --- 1) Load student & tokenizer ---
    print("Loading student model and tokenizer...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_id)
    student = StudentModel(teacher_tokenizer).to(device)
    student.load_state_dict(torch.load(args.student_ckpt, map_location=device))
    student.eval()

    # --- 2) Evaluate on HellaSwag via lm_harness (if requested) ---
    if args.do_hellaswag:
        print("Running HellaSwag evaluation via lm_evaluation_harness...")
        wrapped = StudentCausalLMWrapper(student, teacher_tokenizer, device)

        # Setup args for harness:
        harness_args = {
            "model": wrapped,
            "tasks": ["hellaswag"],                   # HellaSwag task name
            "num_fewshot": 0,                          # zero‐shot
            "batch_size": 1,
            "device": device,                         # for any internal tensor ops
            "no_cache": True,
            "output_dir": None,
        }
        results = lm_eval_run(harness_args)           # returns a dict
        print("HellaSwag results:", results)

    # --- 3) Evaluate on GSM8K via HuggingFace’s `evaluate` metric ---
    if args.do_gsm8k:
        print("Running GSM8K evaluation via HF evaluate...")
        # Load the gsm8k test split
        ds = load_dataset("openai/gsm8k", "main", split="test")
        gsm_metric = hf_evaluate.load("gsm8k", module_type="metric")

        predictions = []
        references  = []

        for example in ds:
            question = example["question"]
            answer   = example["answer"]  # ground‐truth solved expression

            # 3a) Generate student’s answer (string):
            pred_str = student.generate_text([question], max_length=100)[0]
            # 3b) ETL: GSM8K metric expects them as strings, but often you need to post-process
            #     to extract the final numeric result. The metric will handle a “best guess.”
            predictions.append(pred_str)
            references.append(answer)

        # Compute accuracy:
        acc = gsm_metric.compute(predictions=predictions, references=references)
        print("GSM8K accuracy:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student_ckpt",
        type=str,
        required=True,
        help="Path to student_model.pt (the saved state_dict)."
    )
    parser.add_argument(
        "--teacher_model_id",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="The original teacher tokenizer/model ID for vocab."
    )
    parser.add_argument(
        "--do_hellaswag",
        action="store_true",
        help="Whether to run HellaSwag evaluation (via lm_harness)."
    )
    parser.add_argument(
        "--do_gsm8k",
        action="store_true",
        help="Whether to run GSM8K evaluation (via HuggingFace evaluate)."
    )

    args = parser.parse_args()
    evaluate_student(args)