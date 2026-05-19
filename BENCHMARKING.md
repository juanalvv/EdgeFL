# Benchmarking feature README (WIP)

## General idea
To send all benchmarks to a single node for scaling purposes.
Receiving node will have a daemon thread receiving the metrics and sending them to the specific db.

## Questions
* Should we just use a table inside of an already existing db? -> Couldn't on master node

## Currently working parts
Send metrics from all operators into operator1, then save the metric into `benchmarkfl` sqlite db.

### Steps to reproduce
1. In the environment files for the 3 operators add:
```
# Address for benchmarking module, set to point to operator1 node
BENCHMARK_REST_CONN="127.0.0.1:32149"
```

2. Spin up anylog and attach to operator1
3. Run `connect dbms benchmarkfl where type = sqlite and memory = false`
4. Run `get databases` and verify that the benchmarkfl db appears
5. Run some round of training (ingest data, spin up the node servers, use gui...)
6. Verify operators are sending the data -> run `get streaming` in each operator's container
7. In the operator1, run `sql benchmarkfl "select * from fl_benchmarks"`, we should see a list containing one json entry per node per round.
```
{"row_id": 1,
            "insert_timestamp": "2026-05-19 05:18:33.188097",
            "tsd_name": "83",
            "tsd_id": 32,
            "node": "node1",
            "training_index": "b13",
            "round_number": 1,
            "metric_name": "training_time_s",
            "metric_value": 1.5389840602874756,
            "time": 1779167850.5219011},
           {"row_id": 2,
            "insert_timestamp": "2026-05-19 05:18:33.188097",
            "tsd_name": "83",
            "tsd_id": 32,
            "node": "node3",
            "training_index": "b13",
            "round_number": 1,
            "metric_name": "training_time_s",
            "metric_value": 1.5524349212646484,
            "time": 1779167850.522114},
           {"row_id": 3,
            "insert_timestamp": "2026-05-19 05:18:33.188097",
            "tsd_name": "83",
            "tsd_id": 32,
            "node": "node2",
            "training_index": "b13",
            "round_number": 1,
            "metric_name": "training_time_s",
            "metric_value": 1.2555582523345947,
            "time": 1779167852.177545},
```

### Bugs I found while implementing this
- Only node1 appears in the benchmarkfl dbms -> Check if the `BENCHMARK_REST_CONN` is correctly set in the corresponding env file.
    - If the `BENCHMARK_REST_CONN` isn't properly set, the system falls back to EXTERNAL_IP, which means each node will send it to itself, explaining why you're not seeing it appear in operator1's benchmarking dbms.
