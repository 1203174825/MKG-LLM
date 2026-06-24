"""
KG-to-LLM Alignment Layer.

Projects KG encoder output (384-dim) into Qwen embedding space (1536-dim).
The aligned vectors are inserted as soft prompts at special token positions,
enabling the LLM to learn implicit delay patterns, smooth KG noise, and
generalize beyond explicit KG edges.
"""
import torch
import torch.nn as nn
from src.utils.config import CONFIG


class KGAlignmentLayer(nn.Module):
    """Align KG encoder features to Qwen hidden space.
    
    Architecture:
        Stage 1 fused output (384d) → Linear(384→768) → GELU → Dropout
                                    → Linear(768→1024) → GELU → Dropout
                                    → Linear(1024→1536 * n_tokens)
    
    Output: (batch_size, n_tokens, 1536) in Qwen embedding space.
    """
    
    def __init__(self, n_tokens: int = 5):
        super().__init__()
        self.n_tokens = n_tokens
        in_dim = CONFIG.kg.gat_out_dim  # 512
        qwen_dim = 1536  # Qwen2-1.5B hidden size
        
        # Single branch: gradually expand to qwen_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, qwen_dim * n_tokens),
        )
        self.qwen_dim = qwen_dim
    
    def forward(self, kg_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kg_feat: (batch, 512) from Stage 1 KG encoder (fused)
        Returns:
            (batch, n_tokens, 1536) aligned embeddings for Qwen
        """
        batch = kg_feat.shape[0]
        out = self.projector(kg_feat)
        return out.view(batch, self.n_tokens, self.qwen_dim)


class ModifiedEmbedding(nn.Module):
    """Modified Qwen embedding layer that injects KG-aligned soft prompts.
    
    Special token IDs (SIGNAL_TOKEN_ID range) in the input sequence are
    replaced with aligned KG embeddings.
    """
    
    def __init__(self, original_embedding: nn.Embedding, kg_aligner: KGAlignmentLayer):
        super().__init__()
        self.embedding = original_embedding
        self.kg_aligner = kg_aligner
        self.skeleton_aligner = kg_aligner
        
        # Special token range for KG embeddings
        self.signal_token_id = 151925  # Qwen special token range
        
    def forward(self, x: torch.Tensor, kg_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) input token IDs
            kg_feat: (batch, 512) KG features to inject
        Returns:
            (batch, seq_len, 1536) embedding vectors
        """
        if kg_feat is None:
            return self.embedding(x)
            
        B, T = x.size()
        base = self.embedding(x)
        
        # Find positions with special token IDs
        mask = (x >= self.signal_token_id)
        if mask.sum() == 0:
            return base
        
        # Generate aligned embeddings from KG features
        aligned = self.kg_aligner(kg_feat)  # (B, n_tokens, 1536)
        
        # Replace special token positions with aligned embeddings
        for b in range(B):
            b_mask = mask[b]
            num_tokens = int(b_mask.sum().item())
            if num_tokens == 0:
                continue
            # Use the aligned embeddings for the first n_tokens positions
            use_tokens = min(num_tokens, self.kg_aligner.n_tokens)
            indices = b_mask.nonzero(as_tuple=True)[0][:use_tokens]
            base[b, indices, :] = aligned[b, :use_tokens, :]
        
        return base
    
    @property
    def weight(self):
        return self.embedding.weight


class Stage2Model(nn.Module):
    """Full Stage 2 model: Stage1 KG encoder + Alignment Layer + Qwen LLM (LoRA).
    
    The KG encoder weights are frozen. Only the alignment layer and Qwen LoRA
    adapters are trained.
    """
    
    def __init__(self, stage1_ckpt_path: str, qwen_path: str, n_tokens: int = 5):
        super().__init__()
        
        # 1. Load frozen Stage 1 KG encoder
        from src.models.stage1 import Stage1Model
        self.stage1_model = Stage1Model()
        state = torch.load(stage1_ckpt_path, map_location='cpu')
        self.stage1_model.load_state_dict(state['model_state'] if 'model_state' in state else state)
        self.stage1_model.eval()
        for p in self.stage1_model.parameters():
            p.requires_grad = False
        
        # 2. Alignment layer
        self.aligner = KGAlignmentLayer(n_tokens=n_tokens)
        self.n_tokens = n_tokens
        
        # 3. Qwen LLM with modified embedding
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Replace embedding with modified version
        original_embedding = self.qwen_model.get_input_embeddings()
        self.modified_embedding = ModifiedEmbedding(original_embedding, self.aligner)
        self.qwen_model.set_input_embeddings(self.modified_embedding)
        
        self.signal_token_id = 151925
        self.qwen_path = qwen_path
    
    def freeze_qwen_except_lora(self):
        """Freeze all Qwen parameters except LoRA adapters."""
        for name, p in self.qwen_model.named_parameters():
            if 'lora' not in name.lower():
                p.requires_grad = False
            else:
                p.requires_grad = True
        
        # Alignment layer is trainable
        for p in self.aligner.parameters():
            p.requires_grad = True
    
    def forward(self, kg_feat: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None, labels: torch.Tensor = None) -> dict:
        """
        Args:
            kg_feat: (batch, 512) from Stage 1
            input_ids: (batch, seq_len) with special token placeholders
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) for causal LM loss
        Returns:
            dict with loss, logits, etc.
        """
        # Get embeddings with KG features injected
        inputs_embeds = self.modified_embedding(input_ids, kg_feat=kg_feat)
        
        outputs = self.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    def generate(self, kg_feat: torch.Tensor, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor = None,
                 max_new_tokens: int = 32) -> list:
        """Generate text from KG features + prompt."""
        inputs_embeds = self.modified_embedding(input_ids, kg_feat=kg_feat)
        
        outputs = self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # Decode outputs
        decoded = []
        for i in range(outputs.shape[0]):
            out_text = self.tokenizer.decode(outputs[i, input_ids.shape[1]:], skip_special_tokens=True)
            decoded.append(out_text)
        
        return decoded
