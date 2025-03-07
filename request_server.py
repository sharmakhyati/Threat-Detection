import requests

# Define the FastAPI endpoint URL
url = "http://localhost:8000/predict"

# Sample input data in JSON format
sample_data = {
    "id": 1,
    "dur": 0.000011,
    "proto": "udp",
    "service": "-",
    "state": "INT",
    "spkts": 2,
    "dpkts": 0,
    "sbytes": 496,
    "dbytes": 0,
    "rate": 90909.0902,
    "sttl": 254,
    "dttl": 0,
    "sload": 180363632,
    "dload": 0,
    "sloss": 0,
    "dloss": 0,
    "sinpkt": 0.011,
    "dinpkt": 0,
    "sjit": 0,
    "djit": 0,
    "swin": 0,
    "stcpb": 0,
    "dtcpb": 0,
    "dwin": 0,
    "tcprtt": 0,
    "synack": 0,
    "ackdat": 0,
    "smean": 248,
    "dmean": 0,
    "trans_depth": 0,
    "response_body_len": 0,
    "ct_srv_src": 0,
    "ct_state_ttl": 2,
    "ct_dst_ltm": 2,
    "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1,
    "ct_dst_src_ltm": 2,
    "is_ftp_login": 0,
    "ct_ftp_cmd": 0,
    "ct_flw_http_mthd": 0,
    "ct_src_ltm": 1,
    "ct_srv_dst": 2,
    "is_sm_ips_ports": 0
}

# Send a POST request to the FastAPI server
response = requests.post(url, json=sample_data)

# Check the response status and print the prediction
if response.status_code == 200:
    print("Prediction Response:", response.json())
else:
    print(f"Failed to get prediction. Status code: {response.status_code}")
