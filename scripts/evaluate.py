import torch
import argparse
from models import StudentModel, TeacherModel
from distill_utils import get_device
from transformers import AutoTokenizer

# For GSM8K evaluation:
import evaluate as hf_evaluate
from datasets import load_dataset

# For HellaSwag via lm_harness:
from lm_eval.api.model import LM
from lm_eval.evaluator import simple_evaluate
import lm_eval.tasks.hellaswag as Hellaswag

import torch.nn.functional as F

from lm_eval.api.model import LM
import torch

class StudentCausalLMWrapper(LM):
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        self._rank = 0
        self._world_size = 1

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 1024

    @property
    def max_gen_toks(self):
        return 2

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor):
        with torch.no_grad():
            return self.model(inps.to(self.device)).logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            input_ids=context.to(self.device),
            max_new_tokens=max_length,
            eos_token_id=eos_token_id
        )

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            context, continuation = req.args
            # Tokenize context and continuation
            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)

            input_ids = torch.tensor([ctx_ids + cont_ids], device=self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask)  # (1, seq_len, V)

            # Target tokens are next-token shifted
            target_ids = input_ids[:, 1:]
            logits = logits[:, :-1, :]  # Align prediction with target

            # Get logprobs
            logprobs = F.log_softmax(logits, dim=-1)
            selected_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

            # Only score continuation part
            cont_start = len(ctx_ids)
            cont_logprobs = selected_logprobs[0, cont_start:]

            total_logprob = cont_logprobs.sum().item()
            results.append((total_logprob, True))  # True = no truncation

        return results


    def loglikelihood_rolling(self, requests):
        pass

    def generate_until(self, requests):
        # Required for generation tasks like HellaSwag
        results = []
        for context, _ in requests:
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_gen_toks,
                    eos_token_id=self.eot_token_id
                )
            generated_tokens = output_ids[0][input_ids.shape[1]:].tolist()
            results.append(self.tok_decode(generated_tokens))
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
            "limit": 10,
            "device": device,                         # for any internal tensor ops
        }
        results = simple_evaluate(**harness_args)           # returns a dict
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