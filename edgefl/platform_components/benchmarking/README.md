# EdgeFL Benchmarking 

## Overview

EdgeFL runs a federated-learning (FL) lifecycle across multiple training nodes and a single aggregator. The **Benchmarking** subsystem records, for every training round, how the lifecycle behaved: per node, how long each phase took and how accurate the local model became; per aggregator, how long aggregation took and how far apart the nodes' updates arrived. Everything is written to one time-series table (`fl_benchmarks`) that can be queried with SQL.

The subsystem is built to be:

- **Non-invasive** — instrumentation does almost nothing on the training path, and a benchmarking failure can never block or crash the FL loops.
- **Toggle-able** — it can be turned off entirely, per process, with one environment variable.
- **Configurable by environment only** — where metrics are sent, and whether collection is centralized or distributed, are env-var choices that require no code change.

This guide explains how to set it up, how the pieces fit together, and how to read the data back out.

---

## Setup guide

Benchmarking is configured through each node's environment file. The default (and simplest) deployment sends every node's and the aggregator's metrics to the same operator node (operator1 here), which stores them in a local SQLite database.

**1. Add the benchmarking variables to the env file of all three operators *and* the aggregator:**

```
# Send benchmarks to operator1's REST address
BENCHMARK_REST_CONN="127.0.0.1:32149"
# Turn benchmarking on for this process
BENCHMARK_ENABLED="True"
BENCHMARK_FALLBACK=False
```
* Note that the values used here are an example. Point the `BENCHMARK_REST_CONN` to the node where you want the `benchmarkfl` db to live.

**2. Start AnyLog/EdgeLake and attach to the operator1 node.**

**3. Run a round of training** the usual way (ingest data, start the node servers, drive it from the GUI, etc.). If the benchmarker is enabled the metrics will be automatically logged into the db.

**4. Confirm the data landed.** On operator1, query the benchmarking database:

```
sql benchmarkfl "select * from fl_benchmarks"
```

You should see one JSON entry per node per round, plus the aggregator's rows. A quick sanity check:

```
AL operator1 +> sql benchmarkfl "select count(*) from fl_benchmarks where round_number = 1"

{"Query":[{"count(*)":23}],
"Statistics":[{"Count": 1,
                "Time":"00:00:00",
                "Nodes": 1}]}```

No manual database creation is needed — the `benchmarkfl` database is connected automatically on startup if it isn't already present.

### What each flag does

| Variable | Values | Default | What it does |
|---|---|---|---|
| `BENCHMARK_ENABLED` | `True` / `False` | `True` | Master on/off switch for this process. When `False`, the benchmarker is inert: every `record_simple_metric` call returns immediately and nothing is sent or stored. |
| `BENCHMARK_REST_CONN` | `host:port` | *(unset)* | The REST endpoint that should receive this process's metrics — i.e. the ingesting operator. In the default setup, every process sets this to the same operator node (operator1). |
| `BENCHMARKER_FALLBACK` | `True` / `False` | `False` | Only relevant when `BENCHMARK_REST_CONN` is **unset**. If `True`, the process sends metrics to its **own** REST address (`EXTERNAL_IP`) instead of a central collector. This is the opt-in switch for the distributed deployment described in "Deployment topology." |

Two behaviors worth noting:

- **Fallback is never implicit.** Leaving `BENCHMARK_REST_CONN` unset does *not* silently send metrics to the node's own address — that only happens if you explicitly set `BENCHMARKER_FALLBACK=True`.
- **If no destination resolves** (no `BENCHMARK_REST_CONN`, and fallback off), the process logs a clear warning telling you which variable to set and disables benchmarking for itself. It does not crash.

---

## What it measures

Each measurement is stored as **one row**, tagged with the `training_index`, the `round_number`, and the `node` that produced it.

**Per training node** (tagged with the node's name, e.g. `node1`):

| Metric | Meaning |
|---|---|
| `polling_time_s` | Time the node spent waiting for the round to start (idle/polling the blockchain for the start signal). |
| `training_time_s` | Time spent actually training the local model that round. |
| `total_round_time_s` | End-to-end round time for that node (polling + training). |
| `round_accuracy` | Local model accuracy after this round's training. |

**Per aggregator** (tagged with `agg`):

| Metric | Meaning |
|---|---|
| `aggregation_time_s` | Time to combine all nodes' updates into the new global model. |
| `first_to_last_arrival_s` | Spread between the first and last node's update arriving that round — the "straggler gap." |
| `straggling_node_id` | Numeric id of the node whose update arrived last (the round's bottleneck). |

> **`round_accuracy` is reused, not recomputed.** The value is the `final_accuracy` already produced by the post-training inference in `train_model_params` (the accuracy/rollback feature). The benchmarker forwards that number; it does **not** run a second inference pass.

> **`straggling_node_id` is an identifier, not a measurement.** It is stored in the same numeric `metric_value` column as the timings (e.g. `3` for `node3`). The schema is kept uniform on purpose; just remember this particular series is a label when reading it.

---

## Architecture & data flow

The core idea: **instrumentation at the measurement site does almost nothing.** It drops a small record onto an in-memory queue and returns immediately. A **background daemon thread** drains the queue and ships records over the network. This split is what guarantees benchmarking can never block or stall the FL loops.

```
  FL training loop (node_server / aggregator_server)
        │  bench.record_simple_metric(...)   ← returns instantly (just enqueues)
        ▼
   ┌──────────────┐
   │ in-memory    │   thread-safe queue.Queue()
   │  queue       │
   └──────┬───────┘
          │  (background daemon thread, never touches the FL loop)
          ▼
   ┌──────────────┐   HTTP PUT (streaming JSON)
   │ worker thread│ ─────────────────────────────►  AnyLog/EdgeLake Operator REST
   │ requests.    │                                   (ingesting node)
   │  Session     │                                        │
   └──────────────┘                                        ▼
                                                  benchmarkfl (sqlite) → fl_benchmarks table
```

Key properties:

- **Fire-and-forget / non-blocking.** `record_simple_metric` only enqueues. All network I/O happens off the critical path, on the daemon thread. The queue is unbounded, so producers never block; if the sink is unreachable for a long time the queue grows in memory — acceptable at current metric volumes, and a natural place to add a bound later.
- **Failure-isolated — never raises.** If a PUT fails, the worker logs and moves on. A benchmarking outage cannot propagate into training.
- **Connection reuse.** A single `requests.Session` is reused for all writes.
- **Self-provisioning.** On startup the benchmarker checks whether the `benchmarkfl` logical database is connected on the target and connects it (sqlite) if missing — no manual provisioning step at run time.

---

## Code layout

```
edgefl/platform_components/benchmarking/
└── benchmarker.py     ← the Benchmarker class (queue, worker, DB auto-connect, schema headers)

Instrumentation call sites (the only places that touch FL code):
├── node/node_server.py             ← records polling/training/total time + round_accuracy
└── aggregator/aggregator_server.py ← records aggregation time + straggler metrics
```

The `Benchmarker` is instantiated once per server at startup and exposes a single method:

```python
record_simple_metric(training_index, round_number, node_name, metric_name, metric_value, timestamp=None)
```

A **metric allow-list** inside the class rejects unknown `metric_name` values with a warning (typo protection) instead of silently storing garbage. The current allow-list is:

```
training_time_s, polling_time_s, total_round_time_s,
first_to_last_arrival_s, straggling_node_id, round_accuracy, aggregation_time_s
```

---

## The data model

- **Logical database:** `benchmarkfl`
- **Backing store:** SQLite (auto-connected on startup via `connect dbms benchmarkfl where type = sqlite and memory = false`)
- **Table:** `fl_benchmarks`
- **Shape:** *long format* — **one metric per row.**

**Application-written columns:**

| Column | Type | Example |
|---|---|---|
| `node` | string | `node1`, `agg` |
| `training_index` | string | `test-index` |
| `round_number` | int | `7` |
| `metric_name` | string | `training_time_s` |
| `metric_value` | float | `12.34` |
| `time` | float (epoch) | `1780643103.99` |

**System columns added automatically by AnyLog/EdgeLake:** `row_id`, `insert_timestamp`, `tsd_name`, `tsd_id`.

Example rows:

```
node   | training_index | round_number | metric_name        | metric_value
------ | -------------- | ------------ | ------------------ | ---------------
node1  | test-index     | 1            | round_accuracy     | 30.0
node2  | test-index     | 1            | round_accuracy     | 22.0
agg    | test-index     | 1            | aggregation_time_s | 1.84
agg    | test-index     | 1            | straggling_node_id | 3
```

### Why the schema looks like this

Instead of one column per metric (a "wide" table), the data is stored as `(metric_name, metric_value)` pairs — the "long" format. This is a deliberate choice:

- **Adding a metric needs no schema change.** A new metric is just a new `metric_name`; rows appear without any `ALTER TABLE`. In a wide table, every new metric would require a coordinated schema migration across every backing store.
- **One uniform write path.** Every metric — node or aggregator, time or accuracy — shares the same JSON shape and the same code path. No per-metric columns, no NULL padding.
- **The cost** is that queries filter by `metric_name` rather than selecting a named column, and the table is taller. At benchmarking volumes this is negligible.

---

## Deployment topology

The two configuration flags produce two deployment shapes.

### Centralized (default): everything → the same operator node

Every operator **and** the aggregator set `BENCHMARK_REST_CONN` to the same operator node (i.e. all to operator1). All metrics converge into one `benchmarkfl` on that operator.

```
node1 ─┐
node2 ─┼──►  operator1  (benchmarkfl)
node3 ─┤
 agg  ─┘
```

There is one place to query, which keeps analytics simple. The trade-off is that operator1 is the sole writer and carries everyone's benchmark traffic on top of its own FL work.

### Distributed: each node → itself

Leave `BENCHMARK_REST_CONN` unset on the operators and set `BENCHMARKER_FALLBACK=True`. Each operator then writes to its **own local** `benchmarkfl` via `EXTERNAL_IP`.

```
node1 ──► operator1 (benchmarkfl)
node2 ──► operator2 (benchmarkfl)
node3 ──► operator3 (benchmarkfl)
 agg  ──► operator1 (benchmarkfl)   ← MUST stay pointed at op1
```

Write load is spread across operators (it scales with the cluster). The trade-off is that reads now have to fan out across operators rather than hitting a single file (see "Finding the recorded metrics").

Because the storage engine sits behind a logical database name, moving from SQLite to PostgreSQL for higher write volumes is a database-connection change rather than an application rewrite.

### The aggregator exception

The aggregator's own `EXTERNAL_IP` points at the **master** node. The master is a coordination node with **no Operator process**, so it physically cannot ingest streaming data — a PUT there is silently dropped. Therefore:

> **The aggregator must always send to an operator (ie: operator1). It can never "send to itself."**

The aggregator never enables `BENCHMARKER_FALLBACK`. Because fallback is explicit, an unset `BENCHMARK_REST_CONN` on the aggregator cannot silently resolve to the master; instead the aggregator logs a warning naming the variable to set and disables its own benchmarking rather than writing into the void.

> A startup capability probe (`get processes` on the target, which used to refuse non-ingesting nodes) remains in the code **disabled**, as a documented guard should accidental black-holing ever reappear.

---

## Finding the recorded metrics

Metrics are read with standard AnyLog/EdgeLake SQL over REST against the `benchmarkfl` database, on the operator that collected them.

Because of the long format, almost every query **filters by `metric_name`** and groups by `round_number` and/or `node`.

**Look at everything (small runs):**

```
sql benchmarkfl "select * from fl_benchmarks"
```

**Count how many rows a round produced** (one per node per metric, plus aggregator rows):

```
sql benchmarkfl "select count(*) from fl_benchmarks where round_number = 1"
```
* If correctly set up, expect 23 rows per round. 3 metrics from 'agg' node, 20 from operator nodes.

**Training time per node, ordered by round** — useful for spotting a node that slows down over time:

```
sql benchmarkfl "select node, round_number, metric_value from fl_benchmarks where metric_name = 'training_time_s' order by round_number, node"
```

**Accuracy progression for one node:**

```
sql benchmarkfl "select round_number, metric_value from fl_benchmarks where metric_name = 'round_accuracy' and node = 'node1' order by round_number"
```

**Aggregation time per round** (aggregator rows are tagged `agg`):

```
sql benchmarkfl "select round_number, metric_value from fl_benchmarks where node = 'agg' and metric_name = 'aggregation_time_s' order by round_number"
```

**Who was the straggler each round:**

```
sql benchmarkfl "select round_number, metric_value as straggler_node from fl_benchmarks where metric_name = 'straggling_node_id' order by round_number"
```


**Distributed deployments.** When each operator stores its own metrics, a plain `sql` only sees the local node's data. Use AnyLog's network query so it fans out across operators and returns a unified result:

```
run client () sql benchmarkfl "select * from fl_benchmarks where metric_name = 'training_time_s'"
```

---

## Adding a new metric

Adding a metric is intentionally a two-line change, with no migration:

1. **Allow-list it** in `benchmarker.py`:
   ```python
   self.metrics = [..., "peak_memory_mb"]
   ```
2. **Record it** at the relevant instrumentation site:
   ```python
   bench.record_simple_metric(index, round_number, node_name, "peak_memory_mb", value)
   ```

The new rows appear immediately and are queryable by `metric_name = 'peak_memory_mb'`. No `ALTER TABLE`, no coordination across backing stores.

---

## Robustness & safety summary

| Decision | Effect |
|---|---|
| Async queue + daemon worker | Instrumentation never blocks or slows the FL loops. |
| Never raises — logs and continues | A benchmarking fault can't take down training. |
| Explicit `BENCHMARKER_FALLBACK` flag | Self-hosting is opt-in; an unset target never silently aims at `EXTERNAL_IP`, so the aggregator can't accidentally fall back to the non-ingesting master. |
| No resolvable target → disable + warn | The process logs which variable to set and disables benchmarking instead of crashing. |
| `BENCHMARK_ENABLED` toggle | Benchmarking can be turned off entirely per process; every call site becomes a no-op. |
| Metric allow-list | Typos in `metric_name` are caught and warned, not silently stored. |
| Auto-connect the logical DB | `benchmarkfl` is connected on startup if missing — no manual provisioning at run time. |
