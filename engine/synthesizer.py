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
                # logger.error(
                #         f"summarizer_init_failed: {type(e).__name__}: {e}",
                #         exc_info=True
                #     )

    def _build_prompt(self, articles_content: List[str]) -> str:
        combined_texts = []
        for idx, text in enumerate(articles_content):
            trimmed_text = text.strip()[:2000].rsplit(' ', 1)[0]
            if trimmed_text:
                combined_texts.append(f"--- المصدر {idx + 1} ---\n{trimmed_text}")

        joined_articles = "\n\n".join(combined_texts)
        article_count = len(articles_content)

        if article_count >= 3:
            prompt = (
                "<start_of_turn>user\n"
                "أنت محرر صحفي محترف. قم بكتابة ملخص إخباري شامل وموجز بناءً على المصادر المرفقة فقط.\n\n"
                "قواعد صارمة جداً:\n"
                "1. لخص بأسلوبك الخاص. يمنع منعاً باتاً نسخ نصوص أو فقرات طويلة حرفياً من المصادر.\n"
                "2. يجب أن تستخدم نقاط مختصرة (Bullet points) في كل قسم، بحد أقصى 3 نقاط للقسم الواحد.\n"
                "3. لا تضف أي معلومات، تواريخ، أو أسماء غير موجودة في النص.\n"
                "4. إذا لم تجد معلومات كافية لقسم معين، اكتب فقط: 'لا تتوفر معلومات إضافية في المصادر'.\n"
                "5. تجاهل أي روابط أو عبارات ترويجية.\n\n"
                "الأقسام المطلوبة:\n"
                "1. تفاصيل الحدث والتطورات.\n"
                "2. مواقف وتصريحات الأطراف المعنية.\n"
                "3. التحليلات والتبعات المستقبلية.\n\n"
                f"المصادر الإخبارية:\n{joined_articles}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                "التقرير الصحفي الموحد:\n\n"
                "1. تفاصيل الحدث والتطورات:\n"
            )
        else:
            prompt = (
                "<start_of_turn>user\n"
                "أنت محرر صحفي دقيق. قم بكتابة ملخص إخباري قصير ومباشر (في حدود 2-3 أسطر) بناءً على المصدر المرفق.\n"
                "تحذير: لا تنسخ النص حرفياً، لخص بأسلوبك، ولا تخترع أي أسئلة أو معلومات من خارج النص. استخرج الحقائق فقط.\n\n"
                f"المصدر الإخباري:\n{joined_articles}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                "الملخص الإخباري:\n"
            )
        return prompt
    async def synthesize_cluster(self, articles_content: List[str]) -> str:
        if not articles_content:
            return ""

        if not self.is_enabled:
            return "AI Summary is currently unavailable."

        prompt = self._build_prompt(articles_content)
        article_count = len(articles_content)
        
        total_text_length = sum(len(text) for text in articles_content)
        if total_text_length < 150:
            return "المحتوى المتاح قصير جداً (عناوين فقط) ولا يكفي لتوليد ملخص دقيق."

        if self.use_local:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,        
                        do_sample=False,     
                        num_beams=3,              
                        repetition_penalty=1.15,    
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][input_length:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                if article_count >= 3:
                    final_summary = "1. تفاصيل الحدث والتطورات:\n" + decoded
                else:
                    final_summary = decoded

                final_summary = re.sub(r"<[^>]+>", "", final_summary).strip()

                if article_count >= 3 and re.search(r'\n(?:4|10)\.', final_summary):
                    final_summary = re.split(r'\n(?:4|10)\.', final_summary)[0].strip()

                return final_summary
            except Exception as e:
                logger.error("local_gemma_synthesis_failed", error=str(e))
                return ""