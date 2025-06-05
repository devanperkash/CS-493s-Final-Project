from train import train_student
from model_evaluate import evaluate_teacher, evaluate_student

if __name__ == "__main__":
    train = True
    evaluate = True

    if train:
        dataset_examples = 100
        num_epochs = 100
        generation_length = 100
        train_student(False, "hellaswag", dataset_examples, num_epochs, generation_length) # label distillation
        train_student(False, "gsm8k", dataset_examples, num_epochs, generation_length) # label distillation
        train_student(True, "hellaswag", dataset_examples, num_epochs, generation_length) # logits distillation
        train_student(True, "gsm8k", dataset_examples, num_epochs, generation_length) # logits distillation

    if evaluate:
        teacher_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        evaluate_teacher(teacher_model, "hellaswag", eval_examples=30)
        evaluate_teacher(teacher_model, "gsm8k", eval_examples=3)

        student_model = "saved_models/hellaswag_label_student.pt"
        evaluate_student(student_model, teacher_model, "hellaswag", eval_examples=30)

        student_model = "saved_models/hellaswag_logit_student.pt"
        evaluate_student(student_model, teacher_model, "hellaswag", eval_examples=30)

        student_model = "saved_models/gsm8k_label_student.pt"
        evaluate_student(student_model, teacher_model, "gsm8k", eval_examples=3)

        student_model = "saved_models/gsm8k_logit_student.pt"
        evaluate_student(student_model, teacher_model, "gsm8k", eval_examples=3)