from __future__ import annotations
import re

STATEMENT_STR = {0: "reporting", 1: "opinion"}
PROPAGANDA_STR = {0: "neutral", 1: "propaganda"}
ATTRIBUTION_STR = {0: "supported_claim", 1: "unsupported_claim"}

STATEMENT_REASON = {
    0: "النص يركز على نقل وقائع وتفاصيل ملموسة بنبرة حيادية وموضوعية خالية من الانحياز الشخصي.",
    1: "يتضمن النص تعبيرات تدل على التقييم الشخصي، التخمين، أو الرؤية الذاتية للكاتب."
}
PROPAGANDA_REASON = {
    0: "لغة النص تلتزم بالمعايير المهنية والصحفية، وتخلو من الأساليب البلاغية العاطفية أو التشكيك غير المبرر.",
    1: "يلاحظ استخدام صياغات عاطفية، أو محاولات لتوجيه الرأي العام، أو إطلاق أحكام مسبقة."
}
ATTRIBUTION_REASON = {
    0: "الادعاءات الواردة في النص يتم إسنادها بوضوح إلى مصادر محددة، وثائق، أو جهات مسؤولة تدعم صحتها.",
    1: "الادعاءات تُطرح بشكل مرسل دون الإشارة إلى مصادر واضحة، شهادات موثوقة، أو أدلة تدعمها."
}

SYSTEM_PROMPT = (
    "أنت مساعد ذكي متخصص في تصنيف النصوص الإخبارية العربية بدقة. "
    "قم بتحليل العنوان والمحتوى الإخباري وصنفهم لثلاثة مهام:\n"
    "1. نوع العبارة (statement_type): إما reporting أو opinion\n"
    "2. البروباغندا (propaganda): إما neutral أو propaganda\n"
    "3. الإسناد (attribution): إما supported_claim أو unsupported_claim\n\n"
    "قد يتم تزويدك بميزات لغوية محسوبة آلياً مثل نسبة الكلمات والعبارات "
    "المشحونة عاطفياً. استخدم هذه الميزات كإشارات مساعدة ضمن تحليل السياق، "
    "ولا تعتبرها دليلاً حاسماً بمفردها على وجود البروباغندا.\n\n"
    "يجب أن تبدأ إجابتك بخطوة تحليل قصيرة (تحليل النص:)، ثم تكتب التقييم النهائي "
    "بالصيغة التالية بالضبط:\n"
    "التقييم النهائي: [نوع العبارة] | [البروباغندا] | [الإسناد]"
)


def build_user_prompt(title: str, content: str, loaded_words_ratio: float) -> str:
    ratio_percent = loaded_words_ratio * 100
    return (
        f"العنوان: {title}\n"
        f"المحتوى: {content}\n\n"
        f"ميزات لغوية محسوبة آلياً:\n"
        f"- نسبة الكلمات والعبارات المشحونة عاطفياً: {ratio_percent:.4f}%"
    )


def build_assistant_response(st_label: int, pr_label: int, at_label: int, include_reasoning: bool = True) -> str:
    final_line = (
        f"التقييم النهائي: {STATEMENT_STR[st_label]} | {PROPAGANDA_STR[pr_label]} | {ATTRIBUTION_STR[at_label]}"
    )
    if not include_reasoning:
        return final_line
    return (
        f"تحليل النص:\n"
        f"1. نوع العبارة: {STATEMENT_REASON[st_label]}\n"
        f"2. البروباغندا: {PROPAGANDA_REASON[pr_label]}\n"
        f"3. الإسناد: {ATTRIBUTION_REASON[at_label]}\n\n"
        f"{final_line}"
    )


SYSTEM_PROMPT_NO_REASONING = (
    "أنت مساعد ذكي متخصص في تصنيف النصوص الإخبارية العربية بدقة. "
    "قم بتحليل العنوان والمحتوى الإخباري وصنفهم لثلاثة مهام:\n"
    "1. نوع العبارة (statement_type): إما reporting أو opinion\n"
    "2. البروباغندا (propaganda): إما neutral أو propaganda\n"
    "3. الإسناد (attribution): إما supported_claim أو unsupported_claim\n\n"
    "قد يتم تزويدك بميزات لغوية محسوبة آلياً مثل نسبة الكلمات والعبارات "
    "المشحونة عاطفياً. استخدم هذه الميزات كإشارات مساعدة ضمن تحليل السياق.\n\n"
    "أجب بالصيغة التالية بالضبط دون أي نص إضافي:\n"
    "التقييم النهائي: [نوع العبارة] | [البروباغندا] | [الإسناد]"
)


def build_messages(title, content, loaded_words_ratio, st_label=None, pr_label=None, at_label=None,
                    include_answer=True, include_reasoning=True):
    system_content = SYSTEM_PROMPT if include_reasoning else SYSTEM_PROMPT_NO_REASONING
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": build_user_prompt(title, content, loaded_words_ratio)},
    ]
    if include_answer:
        messages.append({
            "role": "assistant",
            "content": build_assistant_response(st_label, pr_label, at_label, include_reasoning=include_reasoning)
        })
    return messages


_LABEL_SETS = {
    "statement_type": {"reporting": 0, "opinion": 1},
    "propaganda": {"neutral": 0, "propaganda": 1},
    "attribution": {"supported_claim": 0, "unsupported_claim": 1},
}

_FINAL_MARKER_RE = re.compile(r"التقييم\s*النهائي\s*[:：]?")
_SPLIT_RE = re.compile(r"[|/\-–—]")


def parse_output(generation_text: str):
    if not generation_text:
        return "unknown", "unknown", "unknown"

    matches = list(_FINAL_MARKER_RE.finditer(generation_text))
    segment = generation_text[matches[-1].end():] if matches else generation_text
    segment = segment.strip().lower()
    segment = segment.splitlines()[0] if segment else segment

    parts = [p.strip().strip(".").replace(" ", "_") for p in _SPLIT_RE.split(segment) if p.strip()]

    def match_label(candidate, label_set):
        if candidate in label_set:
            return candidate
        for key in label_set:
            if key in candidate or candidate in key:
                return key
        return None

    st_pred = match_label(parts[0], _LABEL_SETS["statement_type"]) if len(parts) > 0 else None
    pr_pred = match_label(parts[1], _LABEL_SETS["propaganda"]) if len(parts) > 1 else None
    at_pred = match_label(parts[2], _LABEL_SETS["attribution"]) if len(parts) > 2 else None

    return (st_pred or "unknown"), (pr_pred or "unknown"), (at_pred or "unknown")