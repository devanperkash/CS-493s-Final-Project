from dataloader import load_hellaswag
from distill_utils import get_device, count_parameters
from models import TeacherModel, StudentModel

def train_student(logits_distillation:bool = False):
    # device = get_device()
    # print(f"Running on: {device}")

    hellaswag_subset_train, hellaswag_subset_val = load_hellaswag(5)

    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base")
    student_model = StudentModel(teacher_model.tokenizer)

    logits = teacher_model.get_teacher_y(hellaswag_subset_train['ctx'], max_length=500)

    print(logits)
    

if __name__ == "__main__":
    train_student()

