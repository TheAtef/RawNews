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


    def _strip_leading_hook(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!؟?])\s+', text.strip())
        idx = 0

        while idx < len(sentences) and idx < 2:
            s = sentences[idx].strip()
            if s.endswith('؟') or s.endswith('?') or len(s) < 25:
                idx += 1
            else:
                break
        cleaned = " ".join(sentences[idx:]).strip()
        return cleaned if len(cleaned) > 80 else text.strip()

    def _build_prompt(self, articles_content: List[str]) -> str:
        combined_texts = []
        articles_content = articles_content[:5]
        for idx, text in enumerate(articles_content):
            cleaned = self._strip_leading_hook(text)

            trimmed_text = cleaned[:2800].rsplit(' ', 1)[0]
            if trimmed_text:
                combined_texts.append(f"--- المصدر {idx + 1} ---\n{trimmed_text}")

        joined_articles = "\n\n".join(combined_texts)
        article_count = len(articles_content)

        if article_count >= 3:
            prompt = (
                "<start_of_turn>user\n"
                "أنت محرر صحفي دقيق. مهمتك كتابة ملخص إخباري قصير جداً بناءً على المصادر المرفقة فقط.\n\n"
                "التعليمات الصارمة:\n"
                "1. استخرج الحدث الرئيسي المشترك بين المصادر في جملتين أو 3 جمل فقط، ككتلة نص واحدة متصلة.\n"
                "2. تحذير: يُمنع منعاً باتاً إضافة أي تواريخ أو سنوات، أو خلفيات تاريخية، أو أسماء دول أو جهات أو أطراف (مثل دول أو منظمات دولية) لم تُذكر صراحة في النص أعلاه. لا تنسب أي تصريح أو موقف لأي جهة لم يرد ذكرها حرفياً في المصادر.\n"
                "3. اكتب الملخص بصيغة الخبر (جمل خبرية تقريرية) فقط. يُمنع منعاً باتاً استخدام صيغة السؤال أو علامة الاستفهام (؟) في أي مكان من الملخص، حتى لو وردت أسئلة في المصادر نفسها.\n"
                "4. لا تكرر عناوين المصادر أو أسلوبها الاستفهامي، بل صف ما حدث فعلياً كوقائع.\n"
                "5. يُمنع منعاً باتاً استخدام أي ترقيم أو عناوين فرعية أو نقاط متعددة (مثل 1. أو 2. أو عناوين مثل \"مواقف وأطراف\"). اكتب فقرة واحدة فقط بلا تقسيمات.\n"
                "6. اكتب أسماء الجهات والتنظيمات والأماكن كما وردت بالضبط في النص أعلاه، دون تغيير التهجئة أو اختراع أسماء بديلة.\n"
                "7. توقف عن الكتابة فوراً بمجرد انتهاء التلخيص.\n\n"
                f"المصادر الإخبارية:\n{joined_articles}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                "الملخص الإخباري:\n"
            )
        else:
            prompt = (
                "<start_of_turn>user\n"
                "أنت محرر صحفي دقيق. قم بكتابة ملخص إخباري قصير ومباشر بناءً على المصدر المرفق.\n"
                "تحذير: لا تنسخ النص حرفياً، لخص بأسلوبك، ولا تخترع أي معلومات أو أحداث من خارج النص. استخرج الحقائق فقط.\n"
                "اكتب الملخص بصيغة الخبر فقط، ويُمنع استخدام صيغة السؤال أو علامة الاستفهام (؟).\n\n"
                f"المصدر الإخباري:\n{joined_articles}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                "الملخص الإخباري:\n"
            )
        return prompt

    def _looks_degenerate(self, summary: str) -> bool:
        """Catch outputs that are questions, multi-section reports, or
        contain hallucinated/implausible years rather than a clean 2-3
        sentence summary."""
        if not summary:
            return True
        stripped = summary.strip()

        if stripped.endswith('؟') or stripped.endswith('?'):
            return True
        if stripped.count('؟') + stripped.count('?') >= 1 and len(stripped) < 200:
            return True

        if re.search(r'(^|\n)\s*\d+\s*[.\-:]', stripped):
            return True

        if re.search(r'(^|\n)\s*[ـ\-]\s*(\n|$)', stripped):
            return True

        current_year = 2026
        for match in re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', stripped):
            year = int(match)
            if year < current_year - 5 or year > current_year + 2:
                return True

        if len(stripped) > 600:
            return True

        return False

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
                final_summary = self._generate(prompt, article_count, sample=False)

                # Retry once with light sampling if the greedy pass produced
                # a question/teaser instead of an actual summary.
                if self._looks_degenerate(final_summary):
                    logger.warning("degenerate_summary_retry", cluster_preview=final_summary[:80])
                    final_summary = self._generate(prompt, article_count, sample=True)

                return final_summary
            except Exception as e:
                logger.error("local_gemma_synthesis_failed", error=str(e))
                return ""

    def _generate(self, prompt: str, article_count: int, sample: bool) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        torch.cuda.empty_cache()

        if article_count >= 3:
            if not sample:
                gen_kwargs = {
                    "max_new_tokens": 140,
                    "min_new_tokens": 35,
                    "do_sample": False,
                    "repetition_penalty": 1.12,
                    "no_repeat_ngram_size": 5,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
            else:
  
                gen_kwargs = {
                    "max_new_tokens": 140,
                    "min_new_tokens": 35,
                    "do_sample": True,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "repetition_penalty": 1.12,
                    "no_repeat_ngram_size": 5,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
        else:
            gen_kwargs = {
                "max_new_tokens": 500,
                "do_sample": True,
                "num_beams": 1,
                "temperature": 0.15,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 4,
                "top_p": 0.85,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][input_length:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        decoded = re.sub(r'^[\s\-ـ]+', '', decoded).strip()

        if article_count >= 3:
            final_summary = "- " + decoded
        else:
            final_summary = decoded

        final_summary = re.sub(r"<[^>]+>", "", final_summary).strip()
        final_summary = re.sub(r'(^|\n)\s*[ـ\-]\s*(\n|$)', '\n', final_summary)
        final_summary = re.sub(r'(^|\n)\s*\d+\s*[.\-:]{1,2}\s*[^\n]*?:\s*', '\n', final_summary)
        final_summary = re.sub(r'\n{2,}', '\n', final_summary).strip()
        return final_summary