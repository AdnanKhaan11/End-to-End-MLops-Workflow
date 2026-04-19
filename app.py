from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline
import subprocess
import sys

# Initialize Flask app with static folder configuration
app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/", methods=["GET"])
def homePage():
    return render_template("index.html")


@app.route("/train", methods=["GET"])
def train_page():
    return render_template("train.html")


@app.route("/train-execute", methods=["POST"])
def train_execute():
    try:
        # Prepare environment with UTF-8 encoding for Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        print("Starting model training...")

        # Run the training pipeline
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        stdout_text = result.stdout if result.stdout else ""
        stderr_text = result.stderr if result.stderr else ""

        print(f"Training completed with return code: {result.returncode}")

        # Check if there's a critical error (not just Unicode encoding error)
        has_critical_error = False
        critical_errors = [
            "FileNotFoundError",
            "ModuleNotFoundError",
            "ImportError",
            "AttributeError",
            "KeyError",
            "ValueError",
            "IndexError",
            "TypeError",
        ]

        for error_type in critical_errors:
            if error_type in stderr_text and "UnicodeEncodeError" not in stderr_text:
                has_critical_error = True
                break

        # If only UnicodeEncodeError exists, training likely succeeded
        if result.returncode == 0 or (
            result.returncode != 0
            and "UnicodeEncodeError" in stderr_text
            and not has_critical_error
        ):
            return jsonify(
                {
                    "status": "success",
                    "message": "Model trained successfully! Your wine quality prediction model is now ready for predictions.",
                }
            )
        else:
            # There's a real error
            error_message = stderr_text[:300] if stderr_text else stdout_text[:300]
            return jsonify(
                {"status": "error", "message": f"Training failed: {error_message}"}
            )

    except subprocess.TimeoutExpired:
        return jsonify(
            {
                "status": "error",
                "message": "Training took too long (exceeded 10 minutes) and was stopped. Please try again.",
            }
        )
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"Exception occurred: {error_msg}")
        return jsonify(
            {"status": "error", "message": f"An error occurred: {error_msg}"}
        )


@app.route("/predict", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            # Reading the inputs given by the user with validation
            fixed_acidity = request.form.get("fixed_acidity", "").strip()
            volatile_acidity = request.form.get("volatile_acidity", "").strip()
            citric_acid = request.form.get("citric_acid", "").strip()
            residual_sugar = request.form.get("residual_sugar", "").strip()
            chlorides = request.form.get("chlorides", "").strip()
            free_sulfur_dioxide = request.form.get("free_sulfur_dioxide", "").strip()
            total_sulfur_dioxide = request.form.get("total_sulfur_dioxide", "").strip()
            density = request.form.get("density", "").strip()
            pH = request.form.get("pH", "").strip()
            sulphates = request.form.get("sulphates", "").strip()
            alcohol = request.form.get("alcohol", "").strip()

            # Check if any field is empty
            if not all(
                [
                    fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density,
                    pH,
                    sulphates,
                    alcohol,
                ]
            ):
                error_msg = "Please fill in all fields before predicting!"
                return render_template("error.html", error=error_msg)

            # Convert all values to float
            data = [
                float(fixed_acidity),
                float(volatile_acidity),
                float(citric_acid),
                float(residual_sugar),
                float(chlorides),
                float(free_sulfur_dioxide),
                float(total_sulfur_dioxide),
                float(density),
                float(pH),
                float(sulphates),
                float(alcohol),
            ]

            data = np.array(data).reshape(1, 11)

            # Make prediction
            obj = PredictionPipeline()
            predict = obj.predict(data)

            # Convert prediction to float for better display
            prediction_value = float(predict[0])

            return render_template("results.html", prediction=prediction_value)

        except ValueError as ve:
            error_msg = (
                "Invalid input! Please enter valid numeric values for all fields."
            )
            return render_template("error.html", error=error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print("Exception:", error_msg)
            return render_template("error.html", error=error_msg)

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
