# models.py
from django.db import models
import sqlite3

class Student(models.Model):
    FirstName = models.CharField(max_length=100)
    LastName = models.CharField(max_length=100)
    Gender = models.CharField(max_length=10)
    MedicalCondition = models.CharField(max_length=255)
    Address = models.CharField(max_length=255)
    EmergencyContact = models.CharField(max_length=20)

    def delete(self, *args, **kwargs):
        # Delete from external SQLite database
        conn = sqlite3.connect("sqlite.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM STUDENTS WHERE id=?", (self.id,))
        conn.commit()
        conn.close()
        # Delete from Django database
        super().delete(*args, **kwargs)

class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    medical_condition = models.CharField(max_length=255)
    address = models.CharField(max_length=100)
    emergency_contact = models.CharField(max_length=20)

    def __str__(self):
        return f'{self.first_name} {self.last_name}'
