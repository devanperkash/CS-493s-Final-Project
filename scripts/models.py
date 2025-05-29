from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

class TeacherModel():
    def __init__(self, model_str):
        teacher_model_id = model_str
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)    
        self.model = AutoModelForCausalLM.from_pretrained(teacher_model_id)
        self.model.eval()

    def _run_teacher_model(self, input_text, max_length):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True
            )
        sequences = outputs['sequences']
        logits = outputs['scores']

        return sequences, logits

    def get_teacher_y(self, input_text, max_length=50, logit_distillation:bool = False):
        sequences, logits = self._run_teacher_model(input_text, max_length)

        return logits if logit_distillation else sequences

    def generate_text(self, input_text, max_length=50):
        sequences, _ = self._run_teacher_model(input_text, max_length)

        return self.tokenizer.decode(sequences[0], skip_special_tokens=True)

class StudentModel(nn.Module):
    def __init__(self, teacher_tokenizer, emb_dim=512, n_heads=8, n_layers=4, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = len(teacher_tokenizer)
        self.token_emb = nn.Embedding(self.vocab_size, emb_dim)
        self.tokenizer = teacher_tokenizer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(emb_dim)
        self.output_head = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        logits = self.output_head(x)
        return logits

    def get_student_y_hat(self, input_text, max_length=50):
        pass # TODO

    def generate_text(self, input_text, max_length=50):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        generated_ids = input_ids
        for _ in range(max_length):
            outputs = self.forward(generated_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)