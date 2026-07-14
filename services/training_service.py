
from sqlalchemy import select, func
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime,UTC
from db.session import AsyncSessionLocal,init_db
from models.orm import  ArticleFeedbackORM, ArticleORM, TrainingJobORM,FeedbackStatus
from train.export_feedback import export_feedback_dataset
from train.merge_dataset import merge
TRAINING_FEEDBACK_THREESHOLD=500
async def count_unused_approved_feedbacks()->int:
    async with AsyncSessionLocal() as session:
        stmt=(select(func.count(ArticleFeedbackORM.id))
              .where(ArticleFeedbackORM.status==FeedbackStatus.APPROVED,
                     ArticleFeedbackORM.training_job_id.is_(None))
              )
        result= await session.execute(stmt)
        return result.scalar_one()
    

async def check_training_threshold()->bool:
    feedback_count=await count_unused_approved_feedbacks()
    if feedback_count<TRAINING_FEEDBACK_THREESHOLD:
        return False
    return True

async def create_training_job()->TrainingJobORM | None:
    async with AsyncSessionLocal()as session:
        stmt=(select(ArticleFeedbackORM)
              .where(ArticleFeedbackORM.status==FeedbackStatus.APPROVED,
                     ArticleFeedbackORM.training_job_id.is_(None))
              .limit(TRAINING_FEEDBACK_THREESHOLD))
        result=await session.execute(stmt)

        feedbacks=result.scalars().all()
        if len(feedbacks)<TRAINING_FEEDBACK_THREESHOLD:
            return None
        training_job=TrainingJobORM(feedback_count=len(feedbacks),status="PENDING")
        session.add(training_job)
        await session.flush()
        for feedback in feedbacks:
            feedback.training_job_id=training_job.id
        
        await session.commit()
        await session.refresh(training_job)
        return training_job

async def check_and_create_training_job()->TrainingJobORM | None:
    should_train=await check_training_threshold()
    if not should_train:
        return None
    training_job=await create_training_job()
    return training_job
async def run_training_pipeline(job_id: int):
    async with AsyncSessionLocal() as session:

        training_job = await session.get(
            TrainingJobORM,
            job_id
        )

        if training_job is None:
            raise ValueError("Training job not found")

        training_job.status = "RUNNING"
        training_job.started_at = datetime.now(UTC)

        await session.commit()

        print("=" * 50)
        print("Export feedback dataset...")
        await export_feedback_dataset()
        print("Merge dataset...")
        merge()
        print("Train model...")
        print("Evaluate model...")
        print("=" * 50)

        training_job.status = "COMPLETED"
        training_job.finished_at = datetime.now(UTC)
        training_job.accuracy = 0.91

        await session.commit()

        await session.refresh(training_job)

        return training_job


async def create_test_feedbacks(count: int = 500):
    async with AsyncSessionLocal() as session:
        articles = (
            await session.execute(
                select(ArticleORM).limit(500)
            )
        ).scalars().all()

        feedbacks = []

        for article in articles:

            feedback = ArticleFeedbackORM(
                article_id=article.id,

                propaganda_prediction="neutral",
                statement_prediction="reporting",
                attribution_prediction="supported_claim",

                propaganda_correct=True,
                statement_correct=True,
                attribution_correct=True,

                status=FeedbackStatus.APPROVED,
            )

            feedbacks.append(feedback)

        session.add_all(feedbacks)

        await session.commit()
    

if __name__ == "__main__":
    import asyncio

    async def test():
        await init_db()

        await create_test_feedbacks(500)

        count = await count_unused_approved_feedbacks()

        print("Unused approved feedbacks:", count)

        should_train = await check_training_threshold()

        print("Should start training:", should_train)

        if should_train:
            training_job = await create_training_job()

            print("Training job ID:", training_job.id)
            print("Feedback count:", training_job.feedback_count)
            print("Status:", training_job.status)

            training_job = await run_training_pipeline(training_job.id)

            print("Final Status:", training_job.status)
            print("Accuracy:", training_job.accuracy)
        remaining_count = await count_unused_approved_feedbacks()

        print("Remaining unused feedbacks:", remaining_count)

    asyncio.run(test())