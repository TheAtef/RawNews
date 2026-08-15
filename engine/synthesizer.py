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
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.gemma_model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                )
                self.is_enabled = True
        except Exception as e:
            logger.error("summarizer_init_failed", error=str(e))

    def _build_prompt(self, articles_content: List[str]) -> str:
        combined_texts = []
        for idx, text in enumerate(articles_content):
            trimmed_text = text.strip()[:1800]
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
                        max_new_tokens=700,
                        temperature=0.3,
                        repetition_penalty=1.15,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
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
            payload = {
                "model": settings.gemma_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1024
            }
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(self.api_url, headers=self.headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        logger.error("gemma_api_synthesis_failed", status_code=response.status_code, response=response.text)
                        return ""
            except Exception as e:
                logger.error("gemma_api_synthesis_exception", error=str(e))
                return ""