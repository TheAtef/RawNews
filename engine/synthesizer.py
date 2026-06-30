from __future__ import annotations

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
        

        if self.tokenizer is None or self.model is None:
            logger.info("loading_local_gemma_pipeline", model_id=settings.gemma_model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(settings.gemma_model_id)
            device = "cpu"
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_vram > 8.1: 
                    device = "cuda"

            logger.info("loading_local_gemma_pipeline", model_id=settings.gemma_model_id, target_device=device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.gemma_model_id,
                token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.gemma_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda"
            )

    # def _init_local_pipeline(self) -> None:
    #     if self.tokenizer is None or self.model is None:
    #         logger.info("loading_local_gemma_pipeline", model_id=settings.gemma_model_id)
    #         self.tokenizer = AutoTokenizer.from_pretrained(settings.gemma_model_id)
    #         self.model = AutoModelForCausalLM.from_pretrained(
    #             settings.gemma_model_id,
    #             torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    #             device_map="cuda"
    #         )

    def _build_prompt(self, articles_content: List[str]) -> str:
        combined_texts = ""
        for idx, text in enumerate(articles_content):
            trimmed_text = text.strip()[:1200]
            combined_texts += f"\n--- رواية المصدر {idx + 1} ---\n{trimmed_text}\n"

        prompt = (
            "<start_of_turn>user\n"
            "أنت محرر صحفي محايد. مهمتك هي صياغة تقرير موحد وموضوعي باللغة العربية بناءً على الروايات المختلفة التالية لنفس الحدث.\n"
            "التزم بالقواعد التالية:\n"
            "1. اعتمد فقط على الحقائق المشتركة والمتقاطعة بين المصادر.\n"
            "2. تجنب تبني سردية أي طرف، واكتب بلغة رصينة خالية من العواطف أو الكلمات الانحيازية.\n"
            "3. إذا وجدت تناقضاً جوهرياً بين الروايات دون دليل حاسم، أشر إلى هذا الاختلاف بحيادية تامة دون ترجيح.\n"
            "4. لا تضف أي معلومات خارجية أو استنتاجات شخصية غير واردة في النصوص.\n\n"
            f"النصوص الإخبارية المراد تلخيصها:\n{combined_texts}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "التقرير الصحفي الموحد والحيادي:\n"
        )
        return prompt

    async def synthesize_cluster(self, articles_content: List[str]) -> str:
        if not articles_content:
            return ""

        prompt = self._build_prompt(articles_content)

        if self.use_local:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=800,  
                        temperature=0.2,
                        do_sample=True
                    )
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if "التقرير الصحفي الموحد والحيادي:" in decoded:
                    return decoded.split("التقرير الصحفي الموحد والحيادي:")[-1].strip()
                return decoded.strip()
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