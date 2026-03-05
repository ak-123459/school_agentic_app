import sqlite3
import json
from datetime import datetime
from app.config.config import DB_PATH



conn = sqlite3.connect(DB_PATH)

def create_schema():

    conn.execute("""CREATE TABLE IF NOT EXISTS students (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            dob             TEXT,
            gender          TEXT,
            grade           TEXT,
            class_id        TEXT NOT NULL,
            parent_name     TEXT,
            parent_phone    TEXT,
            parent_email    TEXT,
            address         TEXT,
            enrolled_date   TEXT,
            status          TEXT,    -- active / inactive / graduated
            FOREIGN KEY (class_id)  REFERENCES classes(id)
        )
    """)

    # ── 3. TEACHERS ────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            email       TEXT,
            phone       TEXT,
            subjects    TEXT,        -- JSON list of subjects they teach
            class_id    TEXT NOT  NULL,        -- homeroom class
            joined_date TEXT,
            status      TEXT
        )
    """)

    # ── 4. CLASSES ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id               TEXT PRIMARY KEY,
            name             TEXT,    -- e.g. "9B"
            grade            TEXT,    -- e.g. "9"
            homeroom_teacher TEXT,
            academic_year    TEXT,    -- e.g. "2025-2026"
            capacity         INTEGER,
            FOREIGN KEY (homeroom_teacher) REFERENCES teachers(id)
        )
    """)


    # ── 5. SUBJECTS ────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,  -- Math, Biology, etc.
            grade        TEXT,
            teacher_id   TEXT,
            credit_hours INTEGER,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id)
        )
    """)



    # ── 6. EXAMS ───────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id            TEXT PRIMARY KEY,
            name          TEXT,  -- "Midterm 2026", "Unit Test 1"
            type          TEXT,           -- annual / midterm / unit_test / quiz
            grade         TEXT,
            class_id      TEXT NOT NULL,
            exam_date     TEXT,
            total_marks   INTEGER,
            pass_marks    INTEGER,
            academic_year TEXT,
            FOREIGN KEY (class_id)  REFERENCES classes(id)
        )
    """)

    # ── 7. EXAM TIMETABLE ──────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exam_timetable (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id      TEXT,
            subject_id   TEXT,
            class_id     TEXT NOT NULL,
            grade        TEXT,
            exam_date    TEXT,  
            start_time   TEXT,  -- HH:MM       (NOT NULL)
            end_time     TEXT,    -- HH:MM       (NOT NULL)
            duration_min INTEGER,         -- optional, can be calculated
            room         TEXT,            -- optional, may not be assigned yet
            invigilator  TEXT,            -- optional, may not be assigned yet
            status       TEXT DEFAULT 'scheduled',  -- scheduled / completed / cancelled
            notes        TEXT,            -- optional special instructions
            FOREIGN KEY (exam_id)     REFERENCES exams(id),
            FOREIGN KEY (subject_id)  REFERENCES subjects(id),
            FOREIGN KEY (class_id)    REFERENCES classes(id),
            FOREIGN KEY (invigilator) REFERENCES teachers(id)
        )
    """)

    # ── 8. EXAM RESULTS ────────────────────────────────
    # One row per student per subject per exam
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exam_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT,
            exam_id      TEXT,
            subject_id   TEXT,
            score        REAL,
            max_score    REAL,
            percentage   REAL,
            grade_letter TEXT,
            pass_fail    TEXT,           -- pass / fail
            remarks      TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (exam_id)    REFERENCES exams(id),
            FOREIGN KEY (subject_id) REFERENCES subjects(id)
        )
    """)

    # ── 9. REPORT CARDS ────────────────────────────────
    # Aggregated result per student per exam (all subjects combined)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS report_cards (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT,
            exam_id         TEXT,
            total_score     REAL,
            total_max       REAL,
            percentage      REAL,
            rank            INTEGER,
            grade_letter    TEXT,
            pass_fail       TEXT,
            teacher_remarks TEXT,
            generated_at    TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (exam_id)    REFERENCES exams(id)
        )
    """)

    # ── 10. ATTENDANCE ─────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT,
            class_id    TEXT NOT NULL,
            date        TEXT,
            status      TEXT,           -- present / absent / late / excused
            remarks     TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (class_id)   REFERENCES classes(id)
        )
    """)

    # ── 11. TIMETABLE (regular class schedule) ─────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS timetable (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id    TEXT NOT NULL,
            subject_id  TEXT,
            teacher_id  TEXT,
            day         TEXT,           -- Monday / Tuesday etc.
            start_time  TEXT,
            end_time    TEXT,
            room        TEXT,
            FOREIGN KEY (class_id)   REFERENCES classes(id),
            FOREIGN KEY (subject_id) REFERENCES subjects(id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(id)
        )
    """)

    # ── 12. NOTICES ────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,
            content     TEXT,
            category    TEXT,           -- exam / event / holiday / fee / general / emergency
            target_type TEXT,           -- all / grade / class / student / teacher
            target_id   TEXT,           -- specific id (NULL = everyone)
            created_at  TEXT,
            expires_at  TEXT,           -- when notice becomes inactive
            status      TEXT DEFAULT 'active',  -- active / inactive / archived
            priority    TEXT DEFAULT 'medium'  -- high / medium / low
        )
    """)




    # ── 13. LIBRARY BOOKS ──────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS library_books (
            id        TEXT PRIMARY KEY,
            title     TEXT,
            author    TEXT,
            subject   TEXT,
            grade     TEXT,
            quantity  INTEGER DEFAULT 0,
            available INTEGER DEFAULT 0
        )
    """)

    # ── 14. LIBRARY LOANS ──────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS library_loans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id     TEXT,
            student_id  TEXT,
            issued_date TEXT,
            due_date    TEXT,
            return_date TEXT,           -- NULL until returned
            status      TEXT DEFAULT 'issued',  -- issued / returned / overdue
            FOREIGN KEY (book_id)    REFERENCES library_books(id),
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    # ── 15. EVENTS ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,
            description TEXT,
            event_date  TEXT,
            type        TEXT           -- holiday / sports / exam / trip / ceremony
        )
    """)

    # ── 16. FEES ───────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fees (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT,
            amount      REAL,
            fee_type    TEXT,           -- tuition / transport / library / activity
            due_date    TEXT,
            paid_date   TEXT,           -- NULL until paid
            status      TEXT DEFAULT 'unpaid',  -- paid / unpaid / partial
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    conn.commit()
    print("✅ All tables created successfully!")
    print(f"   📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Show all created tables ─────────────────────────
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cur.fetchall()
    print(f"\n📋 Tables in school.db ({len(tables)} total):")
    for i, (table,) in enumerate(tables, 1):
        print(f"   {i:02}. {table}")


# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    create_schema()