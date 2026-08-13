import os
import shutil

from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal
from models.orm import SummaryFeedbackORM,SummaryTrainingJobORM,FeedbackStatus
SUMMARY_TRAINING_FEEDBACK_THRESHOLD = 500
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