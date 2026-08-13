from __future__ import annotations

import os
import re
import httpx
import torch
import gc
import structlog
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.config import settings

logger = structlog.get_logger(__name__)


class NewsSynthesizer:
    def __init__(self) -> None:
        self.use_local = settings.use_local_gemma_pipeline
        self.tokenizer = None
        self.model = None
        self.is_enabled = False

        try:
            if settings.gemma_model_id.startswith("./") and not os.path.exists(settings.gemma_model_id):
                logger.warning("summarizer_model_not_found", path=settings.gemma_model_id)
                return

            if self.tokenizer is None or self.model is None:
                logger.info("loading_local_gemma_pipeline", model_id=settings.gemma_model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(settings.gemma_model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.gemma_model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    attn_implementation="sdpa" if torch.cuda.is_available() else None
                )
                self.is_enabled = True
        except Exception as e:
            logger.error("summarizer_init_failed", error=str(e))

    def _build_prompt(self, articles_content: List[str]) -> str:
        combined_texts = []
        for idx, text in enumerate(articles_content):
            trimmed_text = text.strip()[:1200]
            if trimmed_text:
                combined_texts.append(f"--- المصدر {idx + 1} ---\n{trimmed_text}")

        joined_articles = "\n\n".join(combined_texts)

        prompt = (
            "<start_of_turn>user\n"
            "أنت محرر صحفي محايد. قم بكتابة تقرير إخباري مفصل وشامل يغطي 3 جوانب رئيسية بناءً على المصادر المعطاة:\n\n"
            "1. تفاصيل الحدث الرئيسي والتطورات الأساسية المذكورة في المصادر.\n"
            "2. مواقف وتصريحات وردود أفعال كافة الأطراف والجهات المعنية.\n"
            "3. التحليلات والتقارير الصحفية المتعلقة بالنواحي الخلافية والتبعات المستقبلية.\n\n"
            f"المصادر الإخبارية:\n{joined_articles}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "التقرير الصحفي الموحد:\n\n"
            "1. تفاصيل الحدث الرئيسي:\n"
        )
        return prompt

    async def synthesize_cluster(self, articles_content: List[str]) -> str:
        if not articles_content:
            return ""

        if not self.is_enabled:
            return "AI Summary is currently unavailable."

        prompt = self._build_prompt(articles_content)

        if self.use_local:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,  
                        do_sample=False,      
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                generated_tokens = outputs[0][input_length:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                final_summary = "1. تفاصيل الحدث الرئيسي:\n" + decoded
                final_summary = re.sub(r"<[^>]+>", "", final_summary).strip()

                return final_summary
            except Exception as e:
                logger.error("local_gemma_synthesis_failed", error=str(e))
                return ""
        else:
            return ""