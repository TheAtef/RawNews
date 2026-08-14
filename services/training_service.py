import shutil
from xml.parsers.expat import model
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime,UTC
from db.session import AsyncSessionLocal,init_db
from models.orm import  ArticleFeedbackORM, ArticleORM, TrainingJobORM,FeedbackStatus
from train.export_feedback import export_feedback_dataset
from train.merge_dataset import merge
from train.train_classifier import run_classifier_training
TRAINING_FEEDBACK_THREESHOLD=1
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

async def get_active_training_job(session:AsyncSession)->TrainingJobORM|None:
        stmt=(select(TrainingJobORM).where(TrainingJobORM.is_active==True))
        result= await session.execute(stmt)
        return result.scalar_one_or_none()
    
def is_new_model_better(active_job,metrics:dict)->bool:
    if active_job is None:
        return True
    return metrics["propaganda_f1"] >= active_job.propaganda_f1
    # return (metrics["propaganda_f1"] >= active_job.propaganda_f1 and metrics["accuracy"] >= active_job.accuracy )


async def rollback_training_job(session: AsyncSession,training_job_id: int,):
    stmt = ( select(ArticleFeedbackORM).where( ArticleFeedbackORM.training_job_id == training_job_id ))

    result = await session.execute(stmt)

    feedbacks = result.scalars().all()

    for feedback in feedbacks:
        feedback.training_job_id = None
            
async def run_training_pipeline(job_id: int):
    async with AsyncSessionLocal() as session:
        training_job = await session.get(TrainingJobORM,job_id)
        if training_job is None:
            raise ValueError("Training job not found")
        training_job.status = "RUNNING"
        training_job.started_at = datetime.now(UTC)
        await session.commit()
        try:
            print("=" * 50)
            print("Export feedback dataset...")
            await export_feedback_dataset()
            print("Merge dataset...")
            merge()
            print("Train model...")
            metrics = run_classifier_training(task_name="propaganda_binary",train_path="train/merged_dataset.jsonl")
            # training_job.accuracy = metrics["accuracy"]
            # training_job.statement_accuracy = metrics["statement_accuracy"]
            # training_job.propaganda_accuracy = metrics["propaganda_accuracy"]
            training_job.propaganda_f1 = metrics["propaganda_f1"]
            # training_job.parse_failure_rate = metrics["parse_failure_rate"]
            # training_job.attribution_accuracy = metrics["attribution_accuracy"]
            training_job.model_path = metrics["model_path"]      
            training_job.status = "COMPLETED"

            print("Evaluate model...")

            old_model=await get_active_training_job(session)

            if is_new_model_better(old_model,metrics):
                if old_model:
                    old_model.is_active=False
                training_job.is_active = True
            await cleanup_old_models(session)


            
            print("=" * 50)
        except Exception as e:
            training_job.status = "FAILED"
            training_job.error_message = str(e)
            await rollback_training_job( session, training_job.id )


        finally:
            training_job.finished_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(training_job)

        return training_job



async def start_retraining_if_needed():
    async with AsyncSessionLocal() as session:
        result=await session.execute(select(TrainingJobORM).where(TrainingJobORM.status.in_(["RUNNING","PENDING"])))
        running_job=result.scalar_one_or_none()
        if running_job is not None:
            print("Training is already running.")

            return None
    
    print("Checking retraining conditions...")
    training_job = await check_and_create_training_job()

    

    if training_job is None:
        print("Not enough feedbacks to start training.")
        return None
    

    training_job = await run_training_pipeline(training_job.id)

    return training_job

async def cleanup_old_models(session: AsyncSession):
    stmt = (select(TrainingJobORM).where(TrainingJobORM.status == "COMPLETED",TrainingJobORM.model_path.is_not(None)).order_by(TrainingJobORM.finished_at.desc()))

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
            print(f"Failed to delete model {model.model_path}:{e}")
if __name__ == "__main__":
    import asyncio
    asyncio.run(start_retraining_if_needed())