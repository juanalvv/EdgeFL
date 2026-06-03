"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""
import traceback
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse

# from dotenv import load_dotenv
from platform_components.EdgeLake_functions.blockchain_EL_functions import get_local_ip, \
    connect_to_db, get_all_databases
from platform_components.node.node import Node
# import numpy as np
import logging
import threading
import time
from dotenv import load_dotenv
import os
import argparse
import requests
import warnings

from uvicorn import run
from fastapi import FastAPI, HTTPException, status

from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from pydantic import BaseModel

from platform_components.lib.logger.logger_config import configure_logging

from platform_components.benchmarking.benchmarker import Benchmarker

warnings.filterwarnings("ignore")

load_dotenv()

edgelake_node_url = f'http://{os.getenv("EXTERNAL_IP")}'
edgelake_node_port = edgelake_node_url.split(":")[2]

configure_logging(f"node_server_{edgelake_node_port}")

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)  # Excludes WARNING, ERROR, CRITICAL

# BENCH - Initialize Benchmarking
_bench_conn = os.getenv("BENCHMARK_REST_CONN") or os.getenv("EXTERNAL_IP")
if not _bench_conn:
    raise RuntimeError("Neither BENCHMARK_REST_CONN nor EXTERNAL_IP is set; cannot init Benchmarker.")

_bench_endpoint = _bench_conn if _bench_conn.startswith("http") else f"http://{_bench_conn}"

_bench_enabled = os.getenv("BENCHMARK_ENABLED", "true").strip().lower() not in ("false", "0", "no", "off")
bench = Benchmarker(_bench_endpoint, enabled=_bench_enabled)

# Initialize the Node instance
node_instance = None
listener_thread = None
stop_listening_thread = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # logger.info("Node server shutting down.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class InitNodeRequest(BaseModel):
    replica_name: str
    replica_ip: str
    replica_port: str
    replica_index: str
    round_number: int


@app.post('/init-node')
def init_node(request: InitNodeRequest):
    global node_instance, listener_thread, stop_listening_thread
    try:
        ip = get_local_ip()
        most_recent_round = request.round_number

        port = request.replica_port
        replica_name = request.replica_name
        index = request.replica_index

        module_name = os.getenv("MODULE_NAME")
        module_file = os.getenv("MODULE_FILE")

        db_name = os.getenv("LOGICAL_DATABASE")

        # Instantiate the Node class
        logger.info(f"{replica_name} before initialized")
        if not node_instance:
            node_instance = Node(replica_name, ip, port, logger)

        if index not in node_instance.databases:
            node_instance.databases[index] = db_name

        node_instance.initialize_specific_node_on_index(index, module_name, module_file)
        node_instance.round_number[index] = most_recent_round # 1 or current round

        logger.info(f"{replica_name} successfully initialized for ({index})")
        # print(f"indexes: {node_instance.indexes}")
        # print(f"module names: {node_instance.module_names}")
        # print(f"module paths: {node_instance.module_paths}")
        # print(f"starting round numbers: {node_instance.round_number}")
        # print(f"training apps: {node_instance.data_handlers}")

        # Start event listener for start round
        listener_thread = threading.Thread(
            name=f"{replica_name}--{index}",
            target=listen_for_start_round,
            args=(node_instance, index, lambda: stop_listening_thread)
        )
        listener_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        listener_thread.start()

        return {
            'status': 'success',
            'message': 'Node initialized successfully'
        }
    except ValueError as e:
        raise ValueError(
            f"No data found in the database: {os.getenv('LOGICAL_DATABASE')}"
        )
    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"/init-node - {str(e)}"
        )
    except ConnectionError as e:
        raise ConnectionError(
            f"Unable to access the database tables: {str(e)}"
        )


def listen_for_start_round(nodeInstance, index, stop_event):
    current_round = nodeInstance.round_number[index]

    logger.info(f"[{index}][Round {current_round}] Listening for start round {current_round}")

    # benchmarking: mark beginning of polling time for this round
    listen_start_ts = time.time()

    while True:
        try:
            headers = {
                'User-Agent': 'AnyLog/1.23',
                'command': f'blockchain get {index} where round_number = {current_round} and node_type = aggregator'
            }
            response = requests.get(edgelake_node_url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                # if no policies, then wait 2 seconds
                if not data:
                    time.sleep(2)
                    continue
                round_data = data[0].get(index)
                # for item in data:
                #     # Check if the key exists in the current dictionary
                #     if f'{index}-r{current_round}' in item:
                #         round_data = item.get(index).get("initParams")
                #         break  # Stop searching once the current round's data is found

                if round_data:
                    logger.debug(f"[{index}] Round Data: {round_data}")  # Debugging line

                    # benchmarking: mark beginning of training time
                    round_start_ts = time.time()
                    #

                    paramsLink = round_data.get('initParams', '')
                    ip_port = round_data.get('ip_port', '')
                    rest_ip_port = round_data.get('rest_ip_port', '')
                    modelUpdate_metadata = nodeInstance.train_model_params(paramsLink, current_round, ip_port, rest_ip_port, index)
                    nodeInstance.add_node_params(current_round, modelUpdate_metadata, index)

                    # benchmarking: mark end of training time
                    round_end_ts = time.time()
                    #

                    logger.info(f"[{index}][Round {current_round}] Step 3 Complete: Model parameters published")
                    
                    # benchmarking
                    training_time_s = round_end_ts - round_start_ts
                    polling_time_s = round_start_ts - listen_start_ts
                    total_round_time_s = round_end_ts - listen_start_ts
                    logger.info(
                            f"Benchmarker [{index}][Round {current_round}] "
                            f" * training time={training_time_s:.3f}s"
                            f" * polling time={polling_time_s:.3f}s"
                            f" * total round time={total_round_time_s:.3f}s"
                    )
                    bench.record_simple_metric(index, current_round, nodeInstance.replica_name, "training_time_s", training_time_s)
                    bench.record_simple_metric(index, current_round, nodeInstance.replica_name, "polling_time_s", polling_time_s)
                    bench.record_simple_metric(index, current_round, nodeInstance.replica_name, "total_round_time_s", total_round_time_s)
                    #

                    current_round += 1
                    logger.info(f"[{index}][Round {current_round}] Listening for start round {current_round}")

                    # benchmarking: reset polling clock for next round
                    listen_start_ts = time.time()

            time.sleep(5)  # Poll every 2 seconds
        except Exception as e:
            logger.error(f"[{index}] Error in listener thread: {str(e)}")
            time.sleep(2)

# TODO: move to a (helper) file (i.e. 'node_helpers.py'?)
# Extracts initParams from the policy 'index-r' at the specified index
def get_most_recent_agg_params(index):
    policy_name = f"{index}-r"
    agg_params = None

    try:
        headers = {
            'User-Agent': 'AnyLog/1.23',
            'command': f'blockchain get {index}'
        }
        response = requests.get(edgelake_node_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            if data:
                policy = data[0]
                policy_data = policy[policy_name]
                agg_params = policy_data["initParams"]

        return agg_params
    except Exception as e:
        logger.error(f"[{index}] Error in extracting round number: {str(e)}")

# @app.route('/inference', methods=['POST'])
@app.post('/inference/{index}', response_class=PlainTextResponse)
def inference(index):
    """Inference on current model w/ data passed in."""
    try:
        logger.info(f"[{index}] received inference request")
        if not index:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Index must be specified."
            )
        results = node_instance.inference(index)
        response = {
                    'index': f'{index}',
                    'status': 'success',
                    'message': 'Inference completed successfully',
                    'model_accuracy': f'{str(results)}'
                    }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

class InferenceRequest(BaseModel):
    input: list
    index: str

# TODO: add index and reformat response to FastAPI PlainTextResponse
# @app.route('/infer', methods=['POST'])
@app.post('/infer')
def direct_inference(request: InferenceRequest):
    """Inference on current model w/ data passed in."""
    try:
        float_list = request.input
        index = request.index
        results = node_instance.direct_inference(index, float_list)
        response = {
            'prediction': str(results),
        }
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error executing inference on model. Check inference function in data handler"
        )

if __name__ == '__main__':
    global port
    parser = argparse.ArgumentParser(description="Run the Node Server.")
    parser.add_argument('--port', type=int, default=8080, help="Port to run the server on.")
    args = parser.parse_args()

    run(
    "node_server:app",
        host="0.0.0.0",
        port=args.port,
        reload=False  # Enable auto-reload on code changes (optional)
    )
