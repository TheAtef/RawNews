from __future__ import annotations

import os
import re
import httpx
import torch
import gc
import structlog
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from core.config import settings

logger = structlog.get_logger(__name__)


class NewsSynthesizer:
    def __init__(self) -> None:
        self.use_local = settings.use_local_gemma_pipeline
        self.tokenizer = None
        self.model = None
        self.is_enabled = False
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.gemma_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa"
)

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
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    attn_implementation="sdpa" if torch.cuda.is_available() else None
                )
                self.is_enabled = True
        except Exception as e:
            logger.error("summarizer_init_failed", error=str(e))

    def _build_prompt_messages(self, valid_articles: List[str]) -> list:
        total_words = sum(len(text.split()) for text in valid_articles)

        if total_words < 50:
            joined = "\n".join([f"- {t.strip()}" for t in valid_articles])
            content = (
                "المعلومات المتوفرة هي مجرد عناوين أو مقتطفات موجزة جداً:\n\n"
                f"{joined}\n\n"
                "المطلوب:\n"
                "اكتب فقرة إخبارية من سطرين فقط تجمع ما ذكر في العناوين أعلاه دون إضافة أية تفاصيل أو افتراضات أو تواريخ من عندك."
            )
            return [{"role": "user", "content": content}]

        if len(valid_articles) == 1:
            content = (
                "أنت محرر إخباري دقيق. مهمتك تلخيص النص المعطى بأمانة تامة.\n\n"
                "القواعد الصارمة:\n"
                "1. اعتمد حصراً على الكلمات والحقائق الواردة في النص.\n"
                "2. يمنع منعاً باتاً تخمين معاني المصطلحات أو إضافة سياقات غير مذكورة.\n"
                "3. لخص الخبر في 2-3 نقاط واضحة ومباشرة.\n\n"
                f"نص الخبر:\n{valid_articles[0]}\n\n"
                "الملخص الإخباري الفعلي:"
            )
            return [{"role": "user", "content": content}]

        sources_text = "\n\n".join([f"[المصدر {i+1}]:\n{text}" for i, text in enumerate(valid_articles)])
        content = (
            "أنت محرر إخباري محايد. قم بتجميع ومقارنة ما ورد في المصادر التالية فقط.\n\n"
            "القواعد الصارمة:\n"
            "1. لا تضف أي حدث أو تاريخ أو تصريح ما لم يكن منصوصاً عليه صراحة في المصادر أدناه.\n"
            "2. اذكر النقاط المشتركة والتصريحات الواردة.\n"
            "3. إذا كانت المصادر مجرد عناوين وتساؤلات، اذكر التساؤلات والمواقف كما هي دون تأليف إجابات أو وقائع.\n\n"
            f"{sources_text}\n\n"
            "التقرير الموحد:"
        )
        return [{"role": "user", "content": content}]

    async def synthesize_cluster(self, articles_content: List[str]) -> str:
        cleaned_articles = [a.strip() for a in articles_content if a and a.strip()]
        if not cleaned_articles:
            return "لا توجد نصوص متوفرة للتلخيص."

        if not self.is_enabled and self.use_local:
            return "AI Summary is currently unavailable."

        messages = self._build_prompt_messages(cleaned_articles)

        if self.use_local:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    prompt = f"<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=700,
                        do_sample=False,            
                        repetition_penalty=1.2,      
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][input_length:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                final_summary = re.sub(r"<[^>]+>", "", decoded).strip()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return final_summary
            except Exception as e:
                logger.error("local_gemma_synthesis_failed", error=str(e))
                return ""
        else:
            payload = {
                "model": settings.gemma_model_id,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 400
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