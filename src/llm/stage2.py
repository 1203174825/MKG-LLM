"""
Stage 2: KG-Constrained LLM Fine-Tuning.
Frozen R-GCN + Alignment + Scaled Residual Injection + Qwen2-1.5B (LoRA) +
3-way Task Attention + Prediction Heads + Semantic Generation.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
from accelerate import Accelerator

from src.models.alignment_injection import (
    AlignmentLayer,
    ScaledResidualInjector,
    KGAlignmentLoss,
)
from src.models.task_attention import TaskAttention
from src.models.prediction_heads import Stage2PredictionHeads
from src.utils.config import CONFIG


class Stage2Model(nn.Module):
    """
    Stage 2: KG-Constrained LLM Fine-Tuning.

    Architecture:
        [Frozen R-GCN] -> Alignment -> Scaled Residual Injection ->
        Qwen2-1.5B (LoRA) -> 3-way Task Attention -> [cls, reg, KG] losses
        + LM head for generation
    """

    def __init__(self, stage1_rgcn=None):
        super().__init__()
        cfg = CONFIG.stage2

        # Frozen R-GCN from Stage 1
        if stage1_rgcn is not None:
            self.rgcn = stage1_rgcn
            for param in self.rgcn.parameters():
                param.requires_grad = False
        else:
            self.rgcn = None  # Will be loaded separately

        # Alignment: 64 -> 1536 + LayerNorm
        self.alignment = AlignmentLayer(kg_dim=64, llm_dim=cfg.alignment_dim)

        # Scaled residual injection
        self.injector = ScaledResidualInjector(
            llm_dim=cfg.alignment_dim,
            lambda_init=cfg.lambda_init,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        )

        # Qwen2-1.5B with LoRA
        self.llm = self._build_llm()
        self.tokenizer = self._build_tokenizer()

        # Special token ID for the "prediction token" (last token)
        # Qwen2 has no [CLS]; we use the last token of the input sequence.
        self.pred_token_id = None  # Will be set based on tokenizer

        # 3-way Task Attention (cls, reg, KG)
        self.task_attention = TaskAttention(
            input_dim=cfg.alignment_dim,
            num_tasks=3,
            hidden_dim=cfg.task_attention_hidden,
        )

        # Prediction heads
        self.pred_heads = Stage2PredictionHeads(input_dim=cfg.alignment_dim)

        # KG alignment loss
        self.kg_loss_fn = KGAlignmentLoss(margin=cfg.loss_KG_margin)

        # Generation loss weight (fixed)
        self.loss_gen_gamma = cfg.loss_gen_gamma

    def _build_llm(self):
        """Load Qwen2-1.5B with LoRA."""
        model_path = CONFIG.paths.llm_weight_dir
        cfg = CONFIG.stage2

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA config
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=PeftTaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Print trainable params
        model.print_trainable_parameters()
        return model

    def _build_tokenizer(self):
        """Load Qwen2 tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.paths.llm_weight_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def forward_for_prediction(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               e_f: torch.Tensor,
                               return_generation: bool = False,
                               gen_max_length: int = 64) -> dict:
        """
        Forward pass for prediction (non-generation head).

        Args:
            input_ids: (batch, seq_len) tokenized input
            attention_mask: (batch, seq_len)
            e_f: (batch, 64) KG embeddings from R-GCN
            return_generation: whether to also generate semantic description
            gen_max_length: max generation length
        Returns:
            dict with prediction outputs
        """
        # 1. Align KG embedding
        e_proj = self.alignment(e_f)  # (batch, 1536)

        # 2. Get LLM input embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # (batch, seq_len, 1536)

        # 3. Scaled residual injection on last token
        inputs_embeds = self.injector(inputs_embeds, e_proj)

        # 4. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last token hidden state for prediction
        h_last = outputs.hidden_states[-1][:, -1, :]  # (batch, 1536)

        # 5. Task attention
        alpha = self.task_attention(h_last)  # (batch, 3)

        # 6. Predictions
        cls_logits, reg_pred = self.pred_heads(h_last)  # (batch, 1), (batch, 1)

        result = {
            "h_last": h_last,
            "alpha": alpha,
            "cls_logits": cls_logits,
            "reg_pred": reg_pred,
            "e_proj": e_proj,
            "logits": outputs.logits,  # for L_gen teacher-forcing
        }

        # 7. Optional: generation
        if return_generation:
            gen_outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_length,
                temperature=CONFIG.stage2.gen_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            gen_text = self.tokenizer.batch_decode(
                gen_outputs[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result["gen_text"] = gen_text

        return result

    def compute_loss(self, cls_logits: torch.Tensor, reg_pred: torch.Tensor,
                     h_last: torch.Tensor, e_f: torch.Tensor,
                     alpha: torch.Tensor,
                     cls_target: torch.Tensor, reg_target: torch.Tensor,
                     gen_logits: torch.Tensor = None,
                     gen_labels: torch.Tensor = None) -> tuple:
        """
        L_stage2 = alpha_cls*BCE + alpha_reg*Huber + alpha_KG*L_KG + gamma*L_gen

        All weights are learned by the single-layer 3-way Task Attention.
        L_gen uses a fixed gamma (not learned).
        """
        bce = self.pred_heads.cls_head.loss(cls_logits, cls_target)
        huber = self.pred_heads.reg_head.loss(reg_pred, reg_target)
        kg_loss = self.kg_loss_fn(h_last, e_f)

        alpha_mean = alpha.mean(dim=0)  # (3,)
        loss = (alpha_mean[0] * bce + alpha_mean[1] * huber +
                alpha_mean[2] * kg_loss)

        if gen_logits is not None and gen_labels is not None:
            # Cross-entropy for generation tokens
            shift_logits = gen_logits[..., :-1, :].contiguous()
            shift_labels = gen_labels[..., 1:].contiguous()
            gen_loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss + self.loss_gen_gamma * gen_loss
        else:
            gen_loss = torch.tensor(0.0)

        return loss, {
            "loss": loss.item(),
            "bce": bce.item(),
            "huber": huber.item(),
            "kg_loss": kg_loss.item(),
            "gen_loss": gen_loss.item() if isinstance(gen_loss, torch.Tensor) else 0.0,
            "alpha_cls": alpha_mean[0].item(),
            "alpha_reg": alpha_mean[1].item(),
            "alpha_kg": alpha_mean[2].item(),
        }
