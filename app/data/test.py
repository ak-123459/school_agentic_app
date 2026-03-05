import os

# ============================================================
# School function keys mapped to vector DB index folders
# ============================================================
SCHOOL_FUNCTION_KEYS = {
    # 📅 SCHEDULE & TIMETABLE
    0:  "get_today_schedule",
    1:  "get_next_class",
    # 📢 ANNOUNCEMENTS
    2:  "get_announcements",
    3:  "get_morning_briefing",
    # 🍽️ CAFETERIA
    4:  "get_lunch_menu",
    # 📝 HOMEWORK & ASSIGNMENTS
    5:  "get_homework",
    6:  "get_assignment_due_date",
    # 🏅 EXAMS & RESULTS
    7:  "get_exam_timetable",
    8:  "get_exam_result",
    9:  "get_grades",
    # 🌤️ HOLIDAYS & EVENTS
    10: "get_holiday_list",
    11: "get_upcoming_events",
    # 👨‍🏫 TEACHERS
    12: "get_teacher_info",
    13: "get_teacher_availability",
    # 🚌 TRANSPORT
    14: "get_bus_schedule",
    15: "get_bus_arrival_time",
    # 📚 LIBRARY
    16: "get_book_availability",
    17: "get_library_timing",
    # 🎓 STUDENT INFO
    18: "get_attendance",
    19: "get_fee_status",
    # 🆘 EMERGENCY
    20: "get_emergency_contacts",
    21: "get_first_aid_location",
    # 🌦️ SCHOOL STATUS
    22: "get_weather_update",
}

# ============================================================
# Base path where all folders will be created
# ============================================================
BASE_PATH = "/Personal-Voice-Agent/langchain_online_RAG/app/data/docs"


def clean_key_name(key_name: str) -> str:
    """Remove 'get_' and 'check_' prefix from key name."""
    for prefix in ["get_"]:
        if key_name.startswith(prefix):
            return key_name[len(prefix):]
    return key_name


def create_directories(base_path: str = BASE_PATH):
    """Create folders for all school function keys."""

    os.makedirs(base_path, exist_ok=True)
    print(f"\n📁 Base directory: {base_path}")
    print("=" * 50)

    created = []
    skipped = []

    for idx, key_name in SCHOOL_FUNCTION_KEYS.items():
        folder_name = clean_key_name(key_name)
        folder_path = os.path.join(base_path, folder_name)

        if os.path.exists(folder_path):
            skipped.append(folder_name)
            print(f"  [{idx:02d}] ⏭️  Already exists : {folder_name}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            created.append(folder_name)
            print(f"  [{idx:02d}] ✅ Created        : {folder_name}")

    print("=" * 50)
    print(f"✅ Created  : {len(created)} folders")
    print(f"⏭️  Skipped  : {len(skipped)} folders")
    print(f"📦 Total    : {len(SCHOOL_FUNCTION_KEYS)} folders")
    print(f"📍 Location : {os.path.abspath(base_path)}\n")


if __name__ == "__main__":
    create_directories(BASE_PATH)