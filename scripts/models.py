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
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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

        self.eos_token_id = teacher_tokenizer.eos_token_id

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
    
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, seq_len) LongTensor of token IDs
        attention_mask: (B, seq_len) BoolTensor or ByteTensor where 1=keep token, 0=pad
        """
        x = self.token_emb(input_ids)  # (B, seq_len, emb_dim)

        # Create a boolean mask for padding: True where pad, False otherwise
        #   src_key_padding_mask expects shape (B, seq_len) with True at padded positions
        if attention_mask is not None:
            # attention_mask from tokenizer is 1 for real tokens, 0 for padded tokens
            src_key_padding_mask = (attention_mask == 0)  # True where pad
        else:
            src_key_padding_mask = None

        # Pass through TransformerEncoder with padding mask
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)                    # (B, seq_len, emb_dim)
        logits = self.output_head(x)        # (B, seq_len, vocab_size)
        return logits

    def get_student_y_hat(self, input_text, max_length=50):
        # ── 1) Tokenize and move to device, *including* attention_mask ──
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        device = next(self.parameters()).device
        input_ids = inputs["input_ids"].to(device)                       # (B, prompt_len)
        attention_mask = inputs.get("attention_mask")                     # (B, prompt_len)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # ── 2) Initialize generation buffers ──
        generated_ids = input_ids                                         # (B, cur_len)
        attn_mask = attention_mask                                        # (B, cur_len)

        B = generated_ids.size(0)
        V = self.vocab_size
        all_logits = torch.empty((B, 0, V), device=device)                # (B, 0, V)

        for _ in range(max_length):
            # Pass current tokens + mask into forward
            outputs = self.forward(generated_ids, attention_mask=attn_mask)  # (B, cur_len, V)
            next_token_logits = outputs[:, -1, :]                            # (B, V)

            all_logits = torch.cat(
                [all_logits, next_token_logits.unsqueeze(1)], dim=1
            )  # (B, t+1, V)

            # Greedy choose next token
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)

            # Append next token ID to generated_ids
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)       # (B, cur_len+1)

            # Append “1” (not padded) to attn_mask for each newly generated token
            if attn_mask is not None:
                # Make a column of ones: shape (B, 1)
                new_mask_col = torch.ones((B, 1), device=device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, new_mask_col], dim=1)            # (B, cur_len+1)
            else:
                # If no initial mask was provided, create one of all-ones
                attn_mask = torch.ones_like(generated_ids, device=device)

        sequences = generated_ids  # (B, prompt_len + max_length)
        logits = all_logits        # (B, max_length, V)

        return sequences, logits
    
    def generate_text(self, input_text, max_length=50):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        device = next(self.parameters()).device
        input_ids = inputs["input_ids"].to(device)                  # (B, prompt_len)
        attention_mask = inputs.get("attention_mask")                # (B, prompt_len)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        generated_ids = input_ids                                      # (B, cur_len)
        attn_mask = attention_mask                                    # (B, cur_len)
        B = input_ids.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length):
            outputs = self.forward(generated_ids, attention_mask=attn_mask)  # (B, cur_len, V)
            next_token_logits = outputs[:, -1, :]                             # (B, V)
            next_token_logits = next_token_logits / 1.5
            for b in range(generated_ids.size(0)):
                for prev_token in generated_ids[b].tolist():
                    next_token_logits[b, prev_token] /= 1.2
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)

            finished |= (next_token_id.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)  # (B, cur_len+1)
            # Expand mask
            if attn_mask is not None:
                new_mask = torch.ones((generated_ids.size(0), 1), device=device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, new_mask], dim=1)
            else:
                attn_mask = torch.ones_like(generated_ids, device=device)

        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
