import asyncio
from db.session import drop_db

async def reset():
    print(" Connecting to PostgreSQL to clean the database...")
    await drop_db()
    print(" All tables dropped and successfully recreated!")

if __name__ == "__main__":
    asyncio.run(reset())