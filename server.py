from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and label encoder
model = xgb.XGBClassifier()
model.load_model('xgboost_model.model')
label_encoder = joblib.load('label_encoder.pkl')

# Define class mapping for reference
class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

# Input data model
class InputData(BaseModel):
    id: int
    dur: float
    proto: str
    service: str
    state: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sttl: int
    dttl: int
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: int
    djit: int
    swin: int
    stcpb: int
    dtcpb: int
    dwin: int
    tcprtt: int
    synack: int
    ackdat: int
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    is_ftp_login: int
    ct_ftp_cmd: int
    ct_flw_http_mthd: int
    ct_src_ltm: int
    ct_srv_dst: int
    is_sm_ips_ports: int

# Root endpoint
@app.get('/')
async def root():
    return {'message': 'Welcome to the XGBoost Inference API!'}

# Get class mapping
@app.get('/class_mapping')
async def get_class_mapping():
    return class_mapping

# Predict endpoint
@app.post('/predict')
async def predict(data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Preprocess categorical features
        categorical_cols = input_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col])

        # Make predictions
        prediction = model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return {
            'predicted_class_int': int(prediction[0]),
            'predicted_class_label': predicted_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
