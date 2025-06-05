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

import datetime
import os

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
        for req in requests:
            context, _ = req.args
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                output_ids = self.generate(
                    input_ids,
                    max_new_tokens=self.max_gen_toks,
                    eos_token_id=self.eot_token_id
                )
            generated_tokens = output_ids[0][input_ids.shape[1]:].tolist()
            results.append(self.tok_decode(generated_tokens))
        return results
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, eos_token_id=None, **kwargs):
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        out_texts = self.model.generate_text(input_texts, max_length=max_new_tokens)
        outputs = self.tokenizer(out_texts, return_tensors="pt", padding=True).input_ids
        return outputs.to(input_ids.device)
    
def evaluate_teacher(teacher_model_id, task, eval_examples):

    eval_out_file = f"evals/teacher_{task}_eval.txt"

    if task == "hellaswag":
        print("Running HellaSwag evaluation on teacher via lm_evaluation_harness...")

        # Setup args for harness:
        harness_args = {
            "model": "hf",  # huggingface model loader
            "model_args": {
                "pretrained": teacher_model_id
            },
            "tasks": ["hellaswag"],                   # HellaSwag task name
            "num_fewshot": 0,                          # zero‐shot
            "batch_size": 8,
            "limit": eval_examples,
            "device": "cuda:0",                         # for any internal tensor ops
        }
        results = simple_evaluate(**harness_args)['results']           # returns a dict
        with open(eval_out_file, 'w') as file:
            file.write(f"Timestamp: {datetime.datetime.now()}")
            file.write(f"\n{results}")
        return results

    # --- 3) Evaluate on GSM8K via HuggingFace’s `evaluate` metric ---
    if task == "gsm8k":
        print("Running GSM8k evaluation on teacher via lm_evaluation_harness...")

        # Setup args for harness:
        harness_args = {
            "model": "hf",  # huggingface model loader
            "model_args": {
                "pretrained": teacher_model_id
            },
            "tasks": ["gsm8k"],                   # GSM8k task name
            "num_fewshot": 0,                          # zero‐shot
            "batch_size": 8,
            "limit": eval_examples,
            "device": "cuda:0",                         # for any internal tensor ops
        }
        results = simple_evaluate(**harness_args)['results']           # returns a dict
        with open(eval_out_file, 'w') as file:
            file.write(f"Timestamp: {datetime.datetime.now()}")
            file.write(f"\n{results}")
        return results
    

def evaluate_student(student_ckpt, teacher_model_id, task, eval_examples):
    device = get_device()

    student_model_name = os.path.splitext(os.path.basename(student_ckpt))[0]
    eval_out_file = f"evals/{student_model_name}_{task}_eval.txt"

    # --- 1) Load student & tokenizer ---
    print("Loading student model and tokenizer...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    student = StudentModel(teacher_tokenizer).to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device))
    student.eval()

    # --- 2) Evaluate on HellaSwag via lm_harness (if requested) ---
    if task == "hellaswag":
        print("Running HellaSwag evaluation via lm_evaluation_harness...")
        wrapped = StudentCausalLMWrapper(student, teacher_tokenizer, device)

        # Setup args for harness:
        harness_args = {
            "model": wrapped,
            "tasks": ["hellaswag"],                   # HellaSwag task name
            "num_fewshot": 0,                          # zero‐shot
            "batch_size": 8,
            "limit": eval_examples,
            "device": "cuda:0",                         # for any internal tensor ops
        }
        results = simple_evaluate(**harness_args)['results']           # returns a dict
        with open(eval_out_file, 'w') as file:
            file.write(f"Timestamp: {datetime.datetime.now()}")
            file.write(f"\n{results}")
        return results

    # --- 3) Evaluate on GSM8K via HuggingFace’s `evaluate` metric ---
    if task == "gsm8k":
        print("Running GSM8k evaluation via lm_evaluation_harness...")
        wrapped = StudentCausalLMWrapper(student, teacher_tokenizer, device)

        # Setup args for harness:
        harness_args = {
            "model": wrapped,
            "tasks": ["gsm8k"],                   # GSM8k task name
            "num_fewshot": 0,                          # zero‐shot
            "batch_size": 8,
            "limit": eval_examples,
            "device": "cuda:0",                         # for any internal tensor ops
        }
        results = simple_evaluate(**harness_args)['results']           # returns a dict
        with open(eval_out_file, 'w') as file:
            file.write(f"Timestamp: {datetime.datetime.now()}")
            file.write(f"\n{results}")
        return results

if __name__ == "__main__":
    teacher_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    results = evaluate_teacher(teacher_model, "hellaswag", 30)
    print(teacher_model, results)

    results = evaluate_teacher(teacher_model, "gsm8k", 3)
    print(teacher_model, results)

    print("-------------------------------------------")

    student_model = "saved_models/hellaswag_label_student.pt"
    results = evaluate_student(student_model, teacher_model, "hellaswag", 30)
    print(student_model, results)

    student_model = "saved_models/hellaswag_logit_student.pt"
    results = evaluate_student(student_model, teacher_model, "hellaswag", 30)
    print(student_model, results)

    student_model = "saved_models/gsm8k_label_student.pt"
    results = evaluate_student(student_model, teacher_model, "gsm8k", 3)
    print(student_model, results)

    student_model = "saved_models/gsm8k_logit_student.pt"
    results = evaluate_student(student_model, teacher_model, "gsm8k", 3)
    print(student_model, results)