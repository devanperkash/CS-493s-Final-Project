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
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True
            )
        sequences = outputs['sequences']
        logits = outputs['scores']

        return sequences, logits

    def get_teacher_y(self, input_text, max_length=None, remove_additional_tokens=True):
        sequences, logits = self._run_teacher_model(input_text, max_length)
        logits = torch.stack(logits, dim=1)[:, :, :len(self.tokenizer)]  if remove_additional_tokens else torch.stack(logits, dim=1) 

        return sequences, logits

    def generate_text(self, input_text, max_length=50):
        sequences, _ = self._run_teacher_model(input_text, max_length)

        return [self.tokenizer.decode(sequence, skip_special_tokens=True) for sequence in sequences]

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
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        device = next(self.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        
        generated_ids = input_ids
        # create an empty tensor of shape (batch_size, 0, vocab_size) on the proper device:
        B = generated_ids.size(0)
        V = self.vocab_size
        all_logits = torch.empty((B, 0, V), device=device)

        for _ in range(max_length):
            outputs = self.forward(generated_ids)           # (B, cur_len, V), on `device`
            next_token_logits = outputs[:, -1, :]           # (B, V)
            all_logits = torch.cat([all_logits, next_token_logits.unsqueeze(1)], dim=1)  # now (B, t, V)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)        # (B, 1)
            generated_ids = torch.cat([generated_ids, next_token_id.to(device)], dim=1) 
        sequences = generated_ids
        logits = all_logits

        return sequences, logits

    def generate_text(self, input_text, max_length=50):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        device = next(self.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        
        generated_ids = input_ids
        for _ in range(max_length):
            outputs = self.forward(generated_ids)              # on `device`
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # on `device`
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)        # stays on `device`

        return [self.tokenizer.decode(sequence, skip_special_tokens=True) for sequence in generated_ids]