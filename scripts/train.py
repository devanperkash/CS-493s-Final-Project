from dataloader import load_hellaswag, load_gsm8k
from distill_utils import get_device, count_parameters
from models import TeacherModel, StudentModel
import torch.nn as nn
import torch.optim as optim

def train_student(logits_distillation:bool = False):
    # device = get_device()
    # print(f"Running on: {device}")

    # Load data
    #subset_train, subset_val = load_hellaswag(3)
    subset_train, subset_val = load_gsm8k(3)

    # Get models setup
    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base") #deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    student_model = StudentModel(teacher_model.tokenizer)

    print('Vocab length:', len(teacher_model.tokenizer))
    print("Size of student model (# of params):")
    count_parameters(student_model)

    # Prep for training
    loss_func = nn.KLDivLoss()
    optimizer = optim.Adam(student_model.parameters())

    # Run models
    for i in range(5):
        t_seqs, t_logits = teacher_model.get_teacher_y(subset_train)
        s_seqs, s_logits = student_model.get_student_y_hat(subset_train)

        optimizer.zero_grad()
        loss = loss_func(t_logits, s_logits)
        loss.backward()
        optimizer.step()

        print(i, loss)

        print("Example:")
        print(teacher_model.generate_text(subset_val[0]))
        print("---")
        print(student_model.generate_text(subset_val[0]))
        print("---")

def test():
    subset_train, subset_val = load_gsm8k(3)

    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base")
    student_model = StudentModel(teacher_model.tokenizer)
    z = teacher_model.generate_text(["What is your name?", "What is 9+9?"])
    print(z)



if __name__ == "__main__":
    #train_student(logits_distillation=True)
    test()

