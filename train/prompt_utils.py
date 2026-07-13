from __future__ import annotations
import re

PROPAGANDA_STR = {0: "neutral", 1: "propaganda"}

PROPAGANDA_REASON = {
    0: "لغة النص تلتزم بالمعايير المهنية والصحفية، وتخلو من الأساليب البلاغية العاطفية أو التشكيك غير المبرر.",
    1: "يلاحظ استخدام صياغات عاطفية، أو محاولات لتوجيه الرأي العام، أو إطلاق أحكام مسبقة."
}

SYSTEM_PROMPT = (
    "أنت مساعد ذكي متخصص في رصد البروباغندا في النصوص الإخبارية العربية بدقة.\n"
    "قم بتحليل العنوان والمحتوى الإخباري وصنفه إلى إحدى الفئتين:\n"
    "1. neutral: نص صحفي موضوعي وخالٍ من لغة البروباغندا.\n"
    "2. propaganda: نص يحتوي على لغة عاطفية مشحونة، تلاعب، أو انحياز.\n\n"
    "قد يتم تزويدك بميزات لغوية محسوبة آلياً مثل نسبة الكلمات والعبارات "
    "المشحونة عاطفياً. استخدم هذه الميزات كإشارات مساعدة ضمن تحليل السياق.\n\n"
    "يجب أن تبدأ إجابتك بخطوة تحليل قصيرة (تحليل النص:)، ثم تكتب التقييم النهائي "
    "بالصيغة التالية بالضبط:\n"
    "التقييم النهائي: [neutral أو propaganda]"
)


def build_user_prompt(title: str, content: str, loaded_words_ratio: float) -> str:
    ratio_percent = loaded_words_ratio * 100
    return (
        f"العنوان: {title}\n"
        f"المحتوى: {content}\n\n"
        f"ميزات لغوية محسوبة آلياً:\n"
        f"- نسبة الكلمات والعبارات المشحونة عاطفياً: {ratio_percent:.4f}%"
    )


def build_assistant_response(pr_label: int, include_reasoning: bool = True) -> str:
    final_line = f"التقييم النهائي: {PROPAGANDA_STR[pr_label]}"
    if not include_reasoning:
        return final_line
    return (
        f"تحليل النص:\n"
        f"البروباغندا: {PROPAGANDA_REASON[pr_label]}\n\n"
        f"{final_line}"
    )


SYSTEM_PROMPT = (
    "أنت مساعد ذكي ومتزن للغاية متخصص في رصد البروباغندا في النصوص الإخبارية العربية بدقة.\n"
    "مهمتك الأساسية هي التمييز الصارم بين التغطية السياسية والبرلمانية الطبيعية وبين البروباغندا الفعلية.\n\n"
    "اتبع القواعد الصارمة التالية لمنع التصنيف الخاطئ:\n"
    "1. التغطية الإخبارية الرسمية للأحداث السياسية، والبرلمانية، والاجتماعات والقرارات الإدارية "
    "تعتبر نصوصاً طبيعية (neutral) كلياً، حتى لو كانت تناقش تطورات حساسة.\n"
    "2. لا تصنف النص كـ (propaganda) إلا إذا احتوى على تلاعب عاطفي فج، لغة تحريضية واضحة، "
    "أو تشويه متعمد ومباشر لعرض الحقائق.\n"
    "3. التقارير الإخبارية الصادرة عن وكالات الأنباء العالمية المعروفة بالحياد (مثل BBC، فرانس 24، العربية) "
    "هي تقارير طبيعية (neutral) كقاعدة عامة، ما لم يحتوي النص بوضوح على انحياز تعبيري صارخ.\n\n"
    "قم بتحليل العنوان والمحتوى الإخباري وصنفه إلى إحدى الفئتين:\n"
    "- neutral\n"
    "- propaganda\n\n"
    "يجب أن تبدأ إجابتك بخطوة تحليل قصيرة (تحليل النص:)، ثم تكتب التقييم النهائي بالصيغة التالية بالضبط:\n"
    "التقييم النهائي: [neutral أو propaganda]"
)


def build_messages(title, content, loaded_words_ratio, pr_label=None,
                    include_answer=True, include_reasoning=True):
    system_content = SYSTEM_PROMPT if include_reasoning else SYSTEM_PROMPT_NO_REASONING
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": build_user_prompt(title, content, loaded_words_ratio)},
    ]
    if include_answer and pr_label is not None:
        messages.append({
            "role": "assistant",
            "content": build_assistant_response(pr_label, include_reasoning=include_reasoning)
        })
    return messages


_FINAL_MARKER_RE = re.compile(r"التقييم\s*النهائي\s*[:：]?")


def parse_output(generation_text: str) -> str:
    if not generation_text:
        return "unknown"

    matches = list(_FINAL_MARKER_RE.finditer(generation_text))
    segment = generation_text[matches[-1].end():] if matches else generation_text
    segment = segment.strip().lower()
    segment = segment.splitlines()[0] if segment else segment

    if "propaganda" in segment:
        return "propaganda"
    elif "neutral" in segment:
        return "neutral"
        
    return "unknown"