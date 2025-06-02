from dataloader import load_hellaswag
from distill_utils import get_device, count_parameters
from models import TeacherModel, StudentModel
import torch.nn as nn
import torch.optim as optim

def train_student(logits_distillation:bool = False):
    # device = get_device()
    # print(f"Running on: {device}")

    # Load data
    hellaswag_subset_train, hellaswag_subset_val = load_hellaswag(3)

    # Get models setup
    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base")
    student_model = StudentModel(teacher_model.tokenizer)

    print('Vocab length:', len(teacher_model.tokenizer))
    print("Size of student model (# of params):")
    count_parameters(student_model)

    # Prep for training
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters())

    # Run models
    for i in range(5):
        t_seqs, t_logits = teacher_model.get_teacher_y(hellaswag_subset_train['ctx'], remove_additional_tokens=True)
        s_seqs, s_logits = student_model.get_student_y_hat(hellaswag_subset_train['ctx'])

        optimizer.zero_grad()
        loss = loss_func(t_logits, s_logits)
        loss.backward()
        optimizer.step()

        print(i, loss)

        print("Example:")
        print("---")
        print(student_model.generate_text(hellaswag_subset_train['ctx']))
        print("---")


if __name__ == "__main__":
    train_student(logits_distillation=True)

