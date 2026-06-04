import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        branch,
        semester,
        attendance_pct,
        internal_marks,
        assignment_score,
        quiz_score,
        study_hours_per_week,
        previous_cgpa,
        activity_participation,
    ):
        self.branch = branch
        self.semester = str(semester)
        self.attendance_pct = float(attendance_pct)
        self.internal_marks = float(internal_marks)
        self.assignment_score = float(assignment_score)
        self.quiz_score = float(quiz_score)
        self.study_hours_per_week = float(study_hours_per_week)
        self.previous_cgpa = float(previous_cgpa)
        self.activity_participation = activity_participation

    def get_data_as_dataframe(self):
        try:
            data = {
                "attendance_pct": [self.attendance_pct],
                "internal_marks": [self.internal_marks],
                "assignment_score": [self.assignment_score],
                "quiz_score": [self.quiz_score],
                "study_hours_per_week": [self.study_hours_per_week],
                "previous_cgpa": [self.previous_cgpa],
                "branch": [self.branch],
                "semester": [self.semester],
                "activity_participation": [self.activity_participation],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)


def get_grade(percentage):
    """Derive letter grade from predicted percentage."""
    if percentage >= 80:
        return "A"
    elif percentage >= 65:
        return "B"
    elif percentage >= 50:
        return "C"
    elif percentage >= 40:
        return "D"
    else:
        return "F"


def get_pass_fail(percentage):
    """Derive pass/fail from predicted percentage."""
    return "Pass" if percentage >= 40 else "Fail"


def get_risk_assessment(percentage, attendance, assignment, quiz, study_hours, cgpa):
    """
    Evaluate student risk level based on multiple factors.
    Returns (risk_level, risk_score, reasons).
    """
    risk_score = 0
    reasons = []

    # ── Attendance check ────────────────────────────────────
    if attendance < 50:
        risk_score += 30
        reasons.append("Very low attendance (below 50%)")
    elif attendance < 65:
        risk_score += 20
        reasons.append("Low attendance (below 65%)")
    elif attendance < 75:
        risk_score += 10
        reasons.append("Attendance below recommended 75%")

    # ── Assignment check ────────────────────────────────────
    if assignment < 40:
        risk_score += 20
        reasons.append("Poor assignment completion (below 40%)")
    elif assignment < 55:
        risk_score += 10
        reasons.append("Below-average assignment scores")

    # ── Quiz check ──────────────────────────────────────────
    if quiz < 40:
        risk_score += 20
        reasons.append("Poor quiz performance (below 40%)")
    elif quiz < 55:
        risk_score += 10
        reasons.append("Below-average quiz scores")

    # ── Study hours check ───────────────────────────────────
    if study_hours < 5:
        risk_score += 15
        reasons.append("Very low study hours (below 5 hrs/week)")
    elif study_hours < 10:
        risk_score += 8
        reasons.append("Low study hours (below 10 hrs/week)")

    # ── CGPA check ──────────────────────────────────────────
    if cgpa < 4.0:
        risk_score += 15
        reasons.append("Very low previous CGPA (below 4.0)")
    elif cgpa < 5.5:
        risk_score += 8
        reasons.append("Below-average previous CGPA")

    # ── Predicted percentage check ──────────────────────────
    if percentage < 40:
        risk_score += 20
        reasons.append("Predicted to FAIL (percentage below 40%)")
    elif percentage < 50:
        risk_score += 10
        reasons.append("Predicted percentage is borderline (below 50%)")

    # Cap at 100
    risk_score = min(risk_score, 100)

    # Determine level
    if risk_score >= 60:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if not reasons:
        reasons.append("Student performance looks healthy across all metrics")

    return risk_level, risk_score, reasons
