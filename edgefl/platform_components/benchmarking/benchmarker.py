import threading
import queue
import logging
import time
import json
import requests


logger = logging.getLogger(__name__)

class Benchmarker:
    def __init__(self, endpoint:str, db_name:str = "benchmarkfl", table_name: str = "fl_benchmarks"):
        self.endpoint = endpoint
        self.q = queue.Queue()
        self.lock = threading.Lock()

        self.state = {} # json pseudo policy with benchmark

        self.session = requests.Session()

        self.metrics = [
                "training_time_s", 
                "wait_time_s", 
                "aggregation_time_s", 
                "straggler_gap_s", 
                "round_accuracy"
            ]

        # table header
        self.header = {
                "type": "json",
                "dbms": db_name,
                "table": table_name,
                "mode": "streaming",
                "Content-Type": "text/plain",
                "User-Agent": "AnyLog/1.23",
        }

        logger.info(f"Benchmarker initialized: \n  endpoint = {self.endpoint}\n  dbms = {db_name}\n  table = {table_name}")

        # DAEMON Queue thread, run in the background asynchronously
        self.worker = threading.Thread(target=self._run_worker, daemon=True)

        self.worker.start()

    def record_simple_metric(self, 
                             training_index, 
                             round_number, 
                             node_name, 
                             metric_name, 
                             metric_value, 
                             timestamp=None):

        if metric_name not in self.metrics:
            logger.warning(f"Benchmarker called with incorrect metric_name value: {metric_name}")
            return


        payload = {
            "node": node_name,
            "training_index": training_index, 
            "round_number": round_number,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "time": time.time() if timestamp is None else timestamp, 
        }

        self.q.put(payload)


    def _run_worker(self):
        while True:
            record = self.q.get()
            try:
                response = self.session.put(
                        self.endpoint, 
                        data=json.dumps(record), 
                        headers=self.header, 
                        timeout=5,
                )

                if response.status_code >= 400:
                    logger.warning(
                        "Bnechmarker POST failed: %s %s", response.status_code, response.text
                    )

            except requests.exceptions.RequestException as e:
                logger.error("Benchmarker PUT error: %s", str(e))
            except Exception as e:
                logger.error("Benchmarker unexpected error: %s", str(e))
            finally:
                self.q.task_done()

