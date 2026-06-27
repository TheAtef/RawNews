
import json
import asyncio

from sqlalchemy import select
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db.session import AsyncSessionLocal
from models.orm import ArticleORM,ArticleFeedbackORM,FeedbackStatus
async def export_feedback_dataset():
    print("Exporter started")
    async with AsyncSessionLocal()as session:
        # result = await session.execute(
        #     select(ArticleORM)
        # )

        # articles = result.scalars().all()

        # print("ARTICLES COUNT =", len(articles))

        # for article in articles:
        #     print(article.id)
        stmt=(select(ArticleORM,ArticleFeedbackORM)
              .join(
            ArticleFeedbackORM,ArticleORM.id==ArticleFeedbackORM.article_id)
            .where(ArticleFeedbackORM.status==FeedbackStatus.APPROVED)
            )
        result=await session.execute(stmt)
        rows=result.all()
        print (rows)
        print(f"Found {len(rows)} approved feedback records")

        exported_rows=[]
        for article,feedback in rows :
            propaganda_label=(feedback.corrected_propaganda 
                              if feedback.propaganda_correct is False and feedback.corrected_propaganda
                              else article.propaganda_label)
            
            statement_type=(feedback.corrected_statement
                            if feedback.statement_correct is False and feedback.corrected_statement
                            else article.statement_type)
            
            attribution_label=(feedback.corrected_attribution
                               if feedback.attribution_correct is False and feedback.corrected_attribution
                               else article.attribution_label)
            record={

                "text" : article.content or "",
                "title": article.title,
                "source":article.source_name,

                "propaganda_label":propaganda_label,
                "statement_type":statement_type,
                "attribution_label":attribution_label,

                "verified":article.verified,
                "reliability_score":article.reliability_score,
                "date":article.published_at.strftime("%Y-%m-%d") if article.published_at else None

            }
            exported_rows.append(record)

            output_path="train/approved_feedback_dataset.jsonl"
            with open(output_path,"w",encoding="utf-8") as f:

                for row in exported_rows:
                    f.write(json.dumps(row,ensure_ascii=False) + "\n")

            print(
                f"Exported {len(exported_rows)} records "
                f"to {output_path}"
            )
            
if __name__ == "__main__":
    asyncio.run(export_feedback_dataset())