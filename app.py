from flask import Flask, request
import sys
import pip
from us_visa.utils.util import read_yaml_file
from matplotlib.style import context
from us_visa.logger import logging
import os, sys
import json
from us_visa.config.configuration import Configuartion
from us_visa.constant import CONFIG_DIR, get_current_time_stamp
from us_visa.pipeline.pipeline import Pipeline
from us_visa.entity.usvisa_approval_predictor import predictor, USVisaData
from flask import send_file, abort, render_template

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "us_visa"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

from us_visa.logger import get_log_dataframe

MODEL_DATA_KEY = "us_visa_data"
PREDICTED_CLASS_KEY = "approval"
PROBABILITY_KEY = "proba"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'us_visa'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("us_visa", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        MODEL_DATA_KEY: None,
        PREDICTED_CLASS_KEY: None,
        PROBABILITY_KEY: None
    }
    if request.method == 'POST':
        continent=request.form["continent"]
        education_of_employee=request.form["education_of_employee"] 
        has_job_experience=request.form["has_job_experience"]
        requires_job_training=request.form["requires_job_training"]
        no_of_employees=int(request.form["no_of_employees"])
        company_age=int(request.form["company_age"])
        region_of_employment=request.form["region_of_employment"]
        prevailing_wage=int(request.form["prevailing_wage"])
        unit_of_wage=request.form["unit_of_wage"]
        full_time_position=request.form["full_time_position"]

        us_visa_data = USVisaData(continent= continent,
                            education_of_employee = education_of_employee,
                            has_job_experience = has_job_experience,
                            requires_job_training = requires_job_training,
                            no_of_employees= no_of_employees,
                            company_age= company_age,
                            region_of_employment = region_of_employment,
                            prevailing_wage= prevailing_wage,
                            unit_of_wage= unit_of_wage,
                            full_time_position= full_time_position,
                            )
        us_visa_df = us_visa_data.get_us_visa_input_data_frame()
        print(us_visa_df)
        visa_predictor = predictor(model_dir=MODEL_DIR)
        predicted_value = visa_predictor.predict(X=us_visa_df)[0]
        if predicted_value==1:
            predicted_message = "The Visa application has low chance of Approval"
        else:
            predicted_message = "The Visa application has high chance of Approval"
            
        probaility = visa_predictor.proba_predict(X=us_visa_df)[0][0]
        probaility = f'Probaility of Visa Approval is {probaility:.2f}%'
        context = {
            MODEL_DATA_KEY: us_visa_data.get_us_visa_data_as_dict(),
            PREDICTED_CLASS_KEY: predicted_message,
            PROBABILITY_KEY: probaility
        }

        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)