"""
Generate a synthetic student performance dataset with realistic correlations.
Output: notebook/data/student_performance_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 2000

# ── Name pools ──────────────────────────────────────────────────
first_names = [
    "Aman", "Priya", "Rahul", "Sneha", "Vikram", "Anjali", "Rohan", "Kavita",
    "Arjun", "Neha", "Deepak", "Pooja", "Karan", "Ritu", "Mohit", "Divya",
    "Aditya", "Shruti", "Nikhil", "Sakshi", "Varun", "Megha", "Harsh", "Tanvi",
    "Siddharth", "Ishita", "Gaurav", "Nisha", "Akash", "Swati", "Rajat", "Simran",
    "Ankit", "Preeti", "Manish", "Komal", "Vivek", "Sonal", "Suresh", "Aarti",
    "Tushar", "Ankita", "Pankaj", "Bhavna", "Kunal", "Jyoti", "Amit", "Rashmi",
]
last_names = [
    "Singh", "Sharma", "Patel", "Kumar", "Gupta", "Verma", "Joshi", "Yadav",
    "Chauhan", "Reddy", "Mishra", "Agarwal", "Thakur", "Pandey", "Dubey",
    "Saxena", "Tiwari", "Nair", "Pillai", "Iyer", "Malhotra", "Khanna",
    "Mehta", "Chopra", "Bhatia", "Srivastava", "Jain", "Kapoor", "Rajput", "Das",
]

branches = ["CSE", "ECE", "ME", "CE", "EE"]
semesters = list(range(1, 9))
activities = ["None", "Low", "Medium", "High"]

# ── Generate features ───────────────────────────────────────────
branch = np.random.choice(branches, N)
semester = np.random.choice(semesters, N)

attendance_pct = np.clip(np.random.normal(72, 18, N), 5, 100).round(1)
internal_marks = np.clip(np.random.normal(28, 12, N), 0, 50).round(1)
assignment_score = np.clip(np.random.normal(62, 22, N), 0, 100).round(1)
quiz_score = np.clip(np.random.normal(58, 22, N), 0, 100).round(1)
study_hours = np.clip(np.random.normal(14, 8, N), 0, 40).round(1)
previous_cgpa = np.clip(np.random.normal(6.2, 1.8, N), 0, 10).round(2)
activity = np.random.choice(activities, N, p=[0.20, 0.30, 0.30, 0.20])

# ── Activity bonus ──────────────────────────────────────────────
activity_bonus = np.where(activity == "High", 3,
                 np.where(activity == "Medium", 1.5,
                 np.where(activity == "Low", 0.5, 0)))

# ── Derive final_percentage (weighted formula) ──────────────────
base = (
    0.20 * attendance_pct
    + 0.25 * (internal_marks / 50 * 100)
    + 0.15 * assignment_score
    + 0.15 * quiz_score
    + 0.10 * (study_hours / 40 * 100)
    + 0.15 * (previous_cgpa / 10 * 100)
)
noise = np.random.normal(0, 5, N)
final_percentage = np.clip(base + noise + activity_bonus, 0, 100).round(2)

# ── Derive targets ──────────────────────────────────────────────
pass_fail = (final_percentage >= 40).astype(int)

def assign_grade(pct):
    if pct >= 80:
        return "A"
    elif pct >= 65:
        return "B"
    elif pct >= 50:
        return "C"
    elif pct >= 40:
        return "D"
    else:
        return "F"

grade = [assign_grade(p) for p in final_percentage]

# ── Names & roll numbers ────────────────────────────────────────
student_names = [
    f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
    for _ in range(N)
]
roll_numbers = [f"{branch[i]}{2024}{str(i + 1).zfill(4)}" for i in range(N)]

# ── Build DataFrame ─────────────────────────────────────────────
df = pd.DataFrame(
    {
        "student_name": student_names,
        "roll_number": roll_numbers,
        "branch": branch,
        "semester": semester,
        "attendance_pct": attendance_pct,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "quiz_score": quiz_score,
        "study_hours_per_week": study_hours,
        "previous_cgpa": previous_cgpa,
        "activity_participation": activity,
        "final_percentage": final_percentage,
        "pass_fail": pass_fail,
        "grade": grade,
    }
)

# ── Save ────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "student_performance_data.csv")
df.to_csv(out_path, index=False)

print(f"[OK] Dataset saved to {out_path}")
print(f"   Rows: {len(df)}  Columns: {list(df.columns)}")
print(f"   Grade distribution:\n{df['grade'].value_counts().sort_index()}")
print(f"   Pass rate: {df['pass_fail'].mean() * 100:.1f}%")
