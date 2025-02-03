import sqlite3
import logging

logger = logging.getLogger(__name__)

def initialize_database(db_path='sqlite.db'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS STUDENTS (
                id INTEGER PRIMARY KEY,
                FirstName TEXT NOT NULL,
                LastName TEXT NOT NULL,
                Gender TEXT NOT NULL,
                MedicalCondition TEXT,
                Address TEXT,
                EmergencyContact TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
