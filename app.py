from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline,
    get_grade,
    get_pass_fail,
    get_risk_assessment,
)
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")

    # Collect form data
    student_name = request.form.get("student_name", "Student")
    roll_number = request.form.get("roll_number", "N/A")

    data = CustomData(
        branch=request.form.get("branch"),
        semester=request.form.get("semester"),
        attendance_pct=float(request.form.get("attendance_pct")),
        internal_marks=float(request.form.get("internal_marks")),
        assignment_score=float(request.form.get("assignment_score")),
        quiz_score=float(request.form.get("quiz_score")),
        study_hours_per_week=float(request.form.get("study_hours_per_week")),
        previous_cgpa=float(request.form.get("previous_cgpa")),
        activity_participation=request.form.get("activity_participation"),
    )

    df = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)
    percentage = round(float(prediction[0]), 2)

    # Clamp to 0-100
    percentage = max(0, min(100, percentage))

    # Derive grade and pass/fail
    grade = get_grade(percentage)
    pass_fail = get_pass_fail(percentage)

    # Risk assessment
    risk_level, risk_score, risk_reasons = get_risk_assessment(
        percentage=percentage,
        attendance=data.attendance_pct,
        assignment=data.assignment_score,
        quiz=data.quiz_score,
        study_hours=data.study_hours_per_week,
        cgpa=data.previous_cgpa,
    )

    results = {
        "student_name": student_name,
        "roll_number": roll_number,
        "percentage": percentage,
        "grade": grade,
        "pass_fail": pass_fail,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_reasons": risk_reasons,
    }

    return render_template("home.html", results=results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running at: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
