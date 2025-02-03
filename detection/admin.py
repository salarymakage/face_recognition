# admin.py
from django.contrib import admin
from .models import Student
import sqlite3

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('id', 'FirstName', 'LastName', 'Gender', 'MedicalCondition', 'Address', 'EmergencyContact')

    def delete_queryset(self, request, queryset):
        # Delete from external SQLite database
        conn = sqlite3.connect("sqlite.db")
        cursor = conn.cursor()
        for student in queryset:
            cursor.execute("DELETE FROM STUDENTS WHERE id=?", (student.id,))
        conn.commit()
        conn.close()
        # Delete from Django database
        queryset.delete()
