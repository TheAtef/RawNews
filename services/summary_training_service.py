import os
import shutil

from datetime import datetime, timezone

from sqlalchemy import select, func
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal ,drop_db,init_db
from models.orm import SummaryFeedbackORM,SummaryTrainingJobORM,FeedbackStatus
from train.export_summary_feedback import export_summary_feedback_dataset
from train.merge_summary_dataset import merge_summary_dataset
from train.train_summarizer import train_gemma_summarizer

SUMMARY_TRAINING_FEEDBACK_THRESHOLD = 1
async def count_unused_approved_summary_feedbacks() -> int:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(func.count(SummaryFeedbackORM.id))
            .where(SummaryFeedbackORM.status == FeedbackStatus.APPROVED,SummaryFeedbackORM.training_job_id.is_(None),)
        )

        result = await session.execute(stmt)
        return result.scalar_one()

async def check_summary_training_threshold() -> bool:
    feedback_count = await count_unused_approved_summary_feedbacks()
    return feedback_count >= SUMMARY_TRAINING_FEEDBACK_THRESHOLD

async def create_summary_training_job() -> SummaryTrainingJobORM | None:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(SummaryFeedbackORM)
            .where(SummaryFeedbackORM.status == FeedbackStatus.APPROVED,SummaryFeedbackORM.training_job_id.is_(None),)
            .limit(SUMMARY_TRAINING_FEEDBACK_THRESHOLD)
        )

        result = await session.execute(stmt)
        feedbacks = result.scalars().all()
        if len(feedbacks) < SUMMARY_TRAINING_FEEDBACK_THRESHOLD:
            return None
        training_job = SummaryTrainingJobORM(feedback_count=len(feedbacks),status="PENDING",)
        session.add(training_job)
        await session.flush()
        for feedback in feedbacks:
            feedback.training_job_id = training_job.id

        await session.commit()
        await session.refresh(training_job)
        return training_job


async def check_and_create_summary_training_job():
    should_train = await check_summary_training_threshold()
    if not should_train:
        return None
    return await create_summary_training_job()


async def get_active_summary_training_job(session: AsyncSession) -> SummaryTrainingJobORM | None:

    stmt = (
        select(SummaryTrainingJobORM)
        .where(SummaryTrainingJobORM.is_active == True)
    )

    result = await session.execute(stmt)

    return result.scalar_one_or_none()


async def cleanup_old_summary_models(session: AsyncSession):
    stmt = (
        select(SummaryTrainingJobORM)
        .where(SummaryTrainingJobORM.status == "COMPLETED",SummaryTrainingJobORM.model_path.is_not(None))
        .order_by(SummaryTrainingJobORM.finished_at.desc())
        )

    result = await session.execute(stmt)
    models = result.scalars().all()
    if len(models) <= 5:
        return
    for model in models[5:]:
        if model.is_active:
            continue

        try:
            if model.model_path and os.path.exists(model.model_path):
                shutil.rmtree(model.model_path)

            model.model_path = None

        except Exception as e:
            print(
                f"Failed to delete model "
                f"{model.model_path}: {e}"
            )

async def rollback_summary_training_job(
    session: AsyncSession,
    training_job_id: int
    ):
    stmt = (
        select(SummaryFeedbackORM)
        .where(SummaryFeedbackORM.training_job_id == training_job_id)
    )

    result = await session.execute(stmt)

    feedbacks = result.scalars().all()

    for feedback in feedbacks:
        feedback.training_job_id = None

async def run_summary_training_pipeline(job_id: int):
    async with AsyncSessionLocal() as session:
        training_job = await session.get(SummaryTrainingJobORM,job_id)

        if training_job is None:
            raise ValueError("Summary training job not found")

        training_job.status = "RUNNING"
        training_job.started_at = datetime.now(timezone.utc)

        await session.commit()

        try:
            print("=" * 50)

            print("Export summary feedback dataset...")
            await export_summary_feedback_dataset()

            print("Merge summary dataset...")
            merge_summary_dataset()

            print("Train summary model...")

            metrics = train_gemma_summarizer(
                dataset_path="train/merged_summary_dataset.jsonl"
            )

            training_job.train_loss = metrics["train_loss"]
            # training_job.eval_loss = metrics["eval_loss"]
            # training_job.model_path = metrics["model_path"]

            training_job.status = "COMPLETED"
            print(f"Model saved at: {training_job.model_path}")


            print("Summary training completed.")
            old_model = await get_active_summary_training_job(session)

            if old_model is None:
                training_job.is_active = True

            await cleanup_old_summary_models(session)

            print("=" * 50)

        except Exception as e:

            training_job.status = "FAILED"
            training_job.error_message = str(e)
            await rollback_summary_training_job(session,training_job.id)
            print(
                f"Summary training failed: {e}"
            )

        finally:

            training_job.finished_at = datetime.now(
                timezone.utc
            )

            await session.commit()
            await session.refresh(training_job)

        return training_job


async def start_summary_retraining_if_needed():

    async with AsyncSessionLocal() as session:

        stmt = (
            select(SummaryTrainingJobORM)
            .where(SummaryTrainingJobORM.status.in_(["RUNNING", "PENDING"]))
            )

        result = await session.execute(stmt)

        running_job = result.scalar_one_or_none()

        if running_job is not None:
            print("Summary training is already running.")
            return None

    print("Checking summary retraining conditions...")
    training_job = await check_and_create_summary_training_job()
    if training_job is None:
        print("Not enough approved summary feedbacks to start training.")
        return None
    print( f"Starting summary training job #{training_job.id}")

    training_job = await run_summary_training_pipeline(training_job.id)

    return training_job


if __name__ == "__main__":
    # import asyncio


    # async def test():

    #     job = await check_and_create_summary_training_job()

    #     if job is None:
    #         print("No training job created.")
    #     else:
    #         print("Training job created:")
    #         print("ID:", job.id)
    #         print("Feedback count:", job.feedback_count)
    #         print("Status:", job.status)

    # asyncio.run(test())
    import asyncio

    asyncio.run(
        run_summary_training_pipeline(1)
    )