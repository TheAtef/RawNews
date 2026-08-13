from __future__ import annotations
import json
import asyncio
import sys
import os
from sqlalchemy import select
from sqlalchemy.orm import selectinload
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db.session import AsyncSessionLocal
from models.orm import ClusterORM, SummaryFeedbackORM, FeedbackStatus
async def export_summary_feedback_dataset():
    print("exporting summary feedback started")
    async with AsyncSessionLocal() as session:
        stmt = (
            select(ClusterORM, SummaryFeedbackORM)
            .options(selectinload(ClusterORM.articles))
            .join(SummaryFeedbackORM,ClusterORM.id == SummaryFeedbackORM.cluster_id)
            .where(SummaryFeedbackORM.status == FeedbackStatus.APPROVED)
            )
        result=await session.execute(stmt)
        rows=result.all()
        print(f"exporting {len(rows)} summary feedbacks")
        exported_rows=[]
        for cluster,feedback in rows:
            summary = (
                feedback.corrected_summary
                if (
                    feedback.user_rating is False
                    and feedback.corrected_summary
                )
                else feedback.generated_summary
            )
            article_text = "\n\n".join(
                            article.content_clean or article.content or article.title
                            for article in cluster.articles
                        )

            if not article_text.strip() or not summary.strip():
                continue
        
            record = {
                "article": article_text,
                "summary": summary,
            }

            exported_rows.append(record)
        output_path = "train/approved_summary_feedback_dataset.jsonl"
        with open(output_path,"w",encoding="utf-8") as f:
            for row in exported_rows:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(f"exported {len(exported_rows)} summary feedbacks to {output_path}")
if __name__=="__main__":
    asyncio.run(export_summary_feedback_dataset())