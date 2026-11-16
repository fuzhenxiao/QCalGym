# Observation and Action Space

This reference describes the Docker release's public calibration interface: **8 API actions**, **48 experiment entry points** across **27 scenes**, and **70 control-field names** across the supported devices. The connected task's `catalog()` and `controls()` responses identify its available functions and exact bounds.

The signatures use Python-style types to describe JSON arguments. `OMITTED` means the key is optional and can be left out of the request. `None` is JSON `null`. Frequencies use GHz, durations use ns, and phases use radians unless a field specifies another unit.

## Contents

- [Transport and Python client](#transport-and-python-client)
- [Public API signatures](#public-api-signatures)
- [Experiment signatures](#experiment-signatures)
- [Control updates](#control-updates)
- [Observation data and artifacts](#observation-data-and-artifacts)

## Transport and Python client

Load the `url` and `token` from the run's `connection.json`. Send requests to `POST /call` with `Content-Type: application/json` and `Authorization: Bearer TOKEN`:

```json
{"action": "run_experiment", "params": {"name": "resonator_spec", "parameters": {"k": 0, "freqs": [7.15, 7.16, 7.17, 7.18, 7.19], "n_avg": 256}}, "request_id": "scan-1"}
```

`action` is one of the eight names below. Its signature describes the contents of `params`. For experiments, put the experiment arguments inside `params.parameters`. For `run_experiment`, `update_calibration` and `submit`, a request ID identifies a receipt: repeating the action with the same ID and arguments returns its stored receipt. Changing the arguments for an existing ID is rejected. These IDs use 1–80 letters, digits, underscores or hyphens. Read-only queries return the requested current state or recorded data.

The convenience methods in [`client.py`](client.py) are:

```python
Client(config: dict | None = None)
Client.call(action: str, params: dict | None = None, request_id: str | None = None) -> dict
Client.run_experiment(name: str, parameters: dict, request_id: str | None = None) -> dict
Client.download(artifact: str) -> bytes
```

If `config` is omitted, `Client` reads the file named by `QCALIBENCH_CONNECTION`. `call()` generates a request ID when none is supplied. `download()` performs an authenticated `GET /artifacts/PATH`. `GET /health` returns `{ "status": "ready" }` without authentication.

## Public API signatures

| Kind | APIs |
| --- | --- |
| Observe state and discover interfaces | `status`, `catalog`, `controls` |
| Acquire observations | `run_experiment` |
| Read recorded observations | `check_raw_data`, `get_data` |
| Apply controls and finish | `update_calibration`, `submit` |

### `status`

```python
status() -> StatusResponse
```

Read current calibration, public objective and remaining budget.

No parameters.

**Returns and behavior:** `{status, state}`. `state` contains `calibration`, `revision`, `budget`, `used`, `objective`, `submitted`, `n_qubits`, `edges`, and, where applicable, `apparatus`. `objective.initial_machine_performance` remains the initial snapshot. Tune objectives also include `parameters_to_fix` with scalar paths and bounds.

### `catalog`

```python
catalog() -> CatalogResponse
```

List all predefined experiments and parameter schemas.

No parameters.

**Returns and behavior:** `{status, api_version, experiments, state}`. `experiments` maps each available name to its description and full JSON parameter schema. Scene-specific limits and experiment availability are reflected in this response.

### `controls`

```python
controls() -> ControlsResponse
```

List all writable calibration fields, exact value schemas, index conventions and coupled constraints.

No parameters.

**Returns and behavior:** `{status, fields, max_updates_per_call, constraints, pair_indices? or flux_ports?}`. Each `fields[name]` supplies `description`, `index` and `value` schemas. Use valid physical pairs from `pair_indices` or ports from `flux_ports`.

### `run_experiment`

```python
run_experiment(name: str, parameters: dict) -> ExperimentReceipt
```

Execute one predefined experiment; obtain fit summaries, figure artifacts and data access handles. Full scan arrays and raw IQ are available on demand via check_raw_data or artifact download.

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `name` | `str` | Required | — |
| `parameters` | `dict` | Required | — |

**Returns and behavior:** `{status, experiment, request_id, fit, meta, figures, point_arrays, fit_curve?, raw_data, response_artifact, cost, state, api_version, data_access}`. `figures` contains `{format, artifact}` handles. Large scan arrays and fitted curves are available through the returned handles. `cost` reports shots, acquisitions, simulated time and host time. Returned state summarizes revision, usage, budget and submission. All experiment entry points below use this receipt format.

### `update_calibration`

```python
update_calibration(updates: list[ControlUpdate]) -> UpdateReceipt
```

Explicitly update calibration fields; does not modify the device model.

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `updates` | `list[dict]` | Required | `{"items": {"type": "object", "properties": {"field": {"type": "string"}, "index": {"oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]}, "value": {}}, "required": ["field", "index", "value"], "additionalProperties": false}}` |

**Returns and behavior:** A receipt with `status`, `request_id`, `state`, `cost`, and recorded artifact handles. Applies 1–64 updates atomically and advances the calibration revision on success. `ControlUpdate = {"field": str, "index": int | list[int], "value": Any}`; the value must match that field's schema. Tune permits changes only to the declared scalar paths. Explicit updates install controls; scan-side `update_cal=True` is disabled in benchmark episodes.

### `check_raw_data`

```python
check_raw_data(
    request_id: str,
    section: str = "manifest",
    array: str | None = None,
    start: int = 0,
    stop: int = 100,
) -> DataResponse
```

Read stored public experiment data without acquiring again. Default manifest lists point names and paginated raw-array/acquisition metadata. points/raw reads a named array slice. fit/meta/state/response explicitly reads a complete recorded section (response can be large). Full JSON and NPZ artifacts are also downloadable. Available after submission.

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `request_id` | `str` | Required | — |
| `section` | `str` | `"manifest"` | `{"enum": ["manifest", "points", "raw", "fit", "meta", "state", "response"]}` |
| `array` | `str \| None` | `null` | — |
| `start` | `int` | `0` | `{"minimum": 0}` |
| `stop` | `int` | `100` | `{"minimum": 1}` |

**Returns and behavior:** `manifest` returns `point_arrays`, `raw_arrays`, acquisition metadata, counts, pagination and artifact handles. `points` returns `{status, array, start, length, values}`. `raw` uses the raw-array slice format. `fit`, `meta`, `state` and `response` return `{status, request_id, SECTION: recorded_value}`. Use `array` for `points` and `raw`. The recorded `state` belongs to that receipt; use `status()` for the current state. Slices satisfy `0 <= start < stop <= start + 2000`.

### `get_data`

```python
get_data(request_id: str, array: str, start: int = 0, stop: int = 2000) -> DataResponse
```

Read up to 2000 samples of an acquired raw IQ array; complex values use {real, imag}. Discover names using check_raw_data first.

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `request_id` | `str` | Required | — |
| `array` | `str` | Required | — |
| `start` | `int` | Optional; omit when unused | `{"minimum": 0}` |
| `stop` | `int` | Optional; omit when unused | `{"minimum": 1}` |

**Returns and behavior:** `{status, array, start, values}`. Reads at most 2,000 samples from a recorded raw array. Discover array names with `check_raw_data(request_id)`. Complex samples use `{real, imag}`. The omitted `start` and `stop` values are 0 and 2000 in the server implementation. Slices satisfy `0 <= start < stop <= start + 2000`.

### `submit`

```python
submit(report: str, claims: dict = OMITTED) -> SubmissionReceipt
```

Finish this episode with a report and claims; receipt does not imply benchmark success.

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `report` | `str` | Required | — |
| `claims` | `dict` | Optional; omit when unused | — |

**Returns and behavior:** A receipt containing `status`, `submission_received`, `note`, `request_id`, `state` and recorded artifact handles. `report` has a maximum length of 16,000 characters; `claims` is optional. Submission ends the episode. Final acceptance and Recovery Score are computed afterwards and written to `final.json` and `result.json` in the run directory.

## Experiment signatures

Call these entries through `client.run_experiment(NAME, PARAMETERS)` or `client.call("run_experiment", {"name": NAME, "parameters": PARAMETERS})`. The environment argument used internally is supplied by QCalGym. Optional defaults shown below come from the public catalog.

An experiment must appear in the connected task's catalog, and its qubit indices, physical edges, installed controls and budget must satisfy the runtime constraints. Named scene groups below preserve the exact schema differences. Benchmark experiments allow at most 441 acquisitions per request; shot and experiment budgets apply separately.

| Entry point | Signature details |
| --- | --- |
| `adc_trace` | [adc_trace](#experiment-adc-trace) |
| `calibrated_sizzle_ramsey` | [calibrated_sizzle_ramsey](#experiment-calibrated-sizzle-ramsey) |
| `cnot_amplitude_scan` | [cnot_amplitude_scan](#experiment-cnot-amplitude-scan) |
| `conditional_ramsey` | [conditional_ramsey](#experiment-conditional-ramsey) |
| `cr_amp_scan` | [cr_amp_scan](#experiment-cr-amp-scan) |
| `cr_entanglement_scan` | [cr_entanglement_scan](#experiment-cr-entanglement-scan) |
| `cr_hamiltonian_tomography` | [cr_hamiltonian_tomography](#experiment-cr-hamiltonian-tomography) |
| `drag_calibration` | [drag_calibration](#experiment-drag-calibration) |
| `drive_phase_scan` | [drive_phase_scan](#experiment-drive-phase-scan) |
| `drive_response` | [drive_response](#experiment-drive-response) |
| `echo` | [echo](#experiment-echo) |
| `flux_iq_acquire` | [flux_iq_acquire](#experiment-flux-iq-acquire) |
| `flux_iq_scan` | [flux_iq_scan](#experiment-flux-iq-scan) |
| `full_xy_scan` | [full_xy_scan](#experiment-full-xy-scan) |
| `gate_fringe` | [gate_fringe](#experiment-gate-fringe) |
| `gate_phase_scan` | [gate_phase_scan](#experiment-gate-phase-scan) |
| `gate_quadratures` | [gate_quadratures](#experiment-gate-quadratures) |
| `gmm3_readout` | [gmm3_readout](#experiment-gmm3-readout) |
| `gmm_readout` | [gmm_readout](#experiment-gmm-readout) |
| `idle_coherence_scan` | [idle_coherence_scan](#experiment-idle-coherence-scan) |
| `idle_response` | [idle_response](#experiment-idle-response) |
| `leakage_vs_pulses` | [leakage_vs_pulses](#experiment-leakage-vs-pulses) |
| `machine_probe` | [machine_probe](#experiment-machine-probe) |
| `mixer_spectrum` | [mixer_spectrum](#experiment-mixer-spectrum) |
| `phase_compensation_scan` | [phase_compensation_scan](#experiment-phase-compensation-scan) |
| `pingpong` | [pingpong](#experiment-pingpong) |
| `power_rabi_amplification` | [power_rabi_amplification](#experiment-power-rabi-amplification) |
| `pump_probe` | [pump_probe](#experiment-pump-probe) |
| `qubit_spec` | [qubit_spec](#experiment-qubit-spec) |
| `rabi_amp` | [rabi_amp](#experiment-rabi-amp) |
| `rabi_phase_scan` | [rabi_phase_scan](#experiment-rabi-phase-scan) |
| `rabi_time` | [rabi_time](#experiment-rabi-time) |
| `ramsey` | [ramsey](#experiment-ramsey) |
| `ramsey_with_tones` | [ramsey_with_tones](#experiment-ramsey-with-tones) |
| `readout_frequency_scan` | [readout_frequency_scan](#experiment-readout-frequency-scan) |
| `readout_joint_scan` | [readout_joint_scan](#experiment-readout-joint-scan) |
| `readout_window_scan` | [readout_window_scan](#experiment-readout-window-scan) |
| `repeated_gate` | [repeated_gate](#experiment-repeated-gate) |
| `resonator_spec` | [resonator_spec](#experiment-resonator-spec) |
| `single_qubit_rb` | [single_qubit_rb](#experiment-single-qubit-rb) |
| `sizzle_zz_scan` | [sizzle_zz_scan](#experiment-sizzle-zz-scan) |
| `spec_vs_power` | [spec_vs_power](#experiment-spec-vs-power) |
| `spectator_echo_timing` | [spectator_echo_timing](#experiment-spectator-echo-timing) |
| `standard_scan` | [standard_scan](#experiment-standard-scan) |
| `stark_shift_scan` | [stark_shift_scan](#experiment-stark-shift-scan) |
| `state_readout` | [state_readout](#experiment-state-readout) |
| `t1` | [t1](#experiment-t1) |
| `xy_plane_cnot_calibration` | [xy_plane_cnot_calibration](#experiment-xy-plane-cnot-calibration) |

<a id="experiment-adc-trace"></a>

### `adc_trace`

```python
adc_trace(k: int, times: list[float], pulse_duration: float = 400.0, shots: int = 1024) -> ExperimentReceipt
```

Acquire adc trace measurements.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `times` | `list[float]` | Required | `{"minItems": 16, "maxItems": 1024, "items": {"type": "number", "minimum": -1000, "maximum": 5000}}` |
| `pulse_duration` | `float` | `400.0` | `{"minimum": 50, "maximum": 2000}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `times` | `list[float]` | Required | `{"minItems": 16, "maxItems": 1024, "items": {"type": "number", "minimum": -1000, "maximum": 5000}}` |
| `pulse_duration` | `float` | `400.0` | `{"minimum": 50, "maximum": 2000}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-calibrated-sizzle-ramsey"></a>

### `calibrated_sizzle_ramsey`

```python
calibrated_sizzle_ramsey(
    i: int,
    j: int,
    delays: list[float],
    virtual_detuning: float = 0.002,
    shots: int = 200,
    spectator: dict | None = None,
    name: str = 'ramsey_tones',
    rise: float | None = None,
) -> ExperimentReceipt
```

Run tone Ramsey using the current stored siZZle settings.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `i` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `j` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `spectator` | `dict \| None` | `null` | `{"additionalProperties": {"type": "integer", "enum": [0, 1]}, "description": "Map qubit index strings to preparation level 0 or 1."}` |
| `name` | `str` | `"ramsey_tones"` | `{"maxLength": 80}` |
| `rise` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `i` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `j` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `spectator` | `dict \| None` | `null` | `{"additionalProperties": {"type": "integer", "enum": [0, 1]}, "description": "Map qubit index strings to preparation level 0 or 1."}` |
| `name` | `str` | `"ramsey_tones"` | `{"maxLength": 80}` |
| `rise` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |

<a id="experiment-cnot-amplitude-scan"></a>

### `cnot_amplitude_scan`

```python
cnot_amplitude_scan(amps: list[float], shots: int = 1024) -> ExperimentReceipt
```

Installed CNOT amplitude response for 00/ZZ, +0/XX and ++/XX. Restores amplitude; three acquisitions per point.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `amps` | `list[float]` | Required | `{"items": {"type": "number", "minimum": -2, "maximum": 2}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-conditional-ramsey"></a>

### `conditional_ramsey`

```python
conditional_ramsey(
    k: int,
    spectator: int,
    delays: list[float],
    virtual_detuning: float = 0.002,
    shots: int = 200,
) -> ExperimentReceipt
```

Ramsey traces with the spectator prepared in 0 and in 1; fitted frequencies are unsigned.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `spectator` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `spectator` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-cr-amp-scan"></a>

### `cr_amp_scan`

```python
cr_amp_scan(
    control: int,
    target: int,
    amps: list[float],
    duration: float | None = None,
    phase: float | None = None,
    shots: int = 200,
) -> ExperimentReceipt
```

Scan CR amplitude for control in 0/1 and fit target oscillations.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `amps` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `duration` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |
| `phase` | `float \| None` | `null` | `{"minimum": -20, "maximum": 20}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `amps` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `duration` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |
| `phase` | `float \| None` | `null` | `{"minimum": -20, "maximum": 20}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-cr-entanglement-scan"></a>

### `cr_entanglement_scan`

```python
cr_entanglement_scan(control: int, target: int, amps: list[float], shots: int = 1024, repetitions: int = 1) -> ExperimentReceipt
```

Measure XYZ target Bloch vectors after one or three raw CR pulses, control prepared in 0/1. No gate fidelity returned.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |
| `amps` | `list[float]` | Required | `{"minItems": 5, "maxItems": 73, "items": {"type": "number", "minimum": 0, "maximum": 2}}` |
| `repetitions` | `int` | `1` | `{"enum": [1, 3]}` |

<a id="experiment-cr-hamiltonian-tomography"></a>

### `cr_hamiltonian_tomography`

```python
cr_hamiltonian_tomography(
    control: int,
    target: int,
    durations: list[float],
    amp: float | None = None,
    phase: float | None = None,
    shots: int = 200,
    rise: float | None = None,
) -> ExperimentReceipt
```

Measure standard Pauli target XYZ versus CR duration for both control states; fit Bloch vectors w0,w1 in rad/ns. I*=(w1+w0)/(4pi), Z*=(w1-w0)/(4pi) in GHz; Z* uses control1-minus-control0, opposite the standard control-Z convention. For undriven diagonal terms, conditional transition shift = -2*ZZ_GHz.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `durations` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `amp` | `float \| None` | `null` | `{"minimum": -2, "maximum": 2}` |
| `phase` | `float \| None` | `null` | `{"minimum": -20, "maximum": 20}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `rise` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `durations` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `amp` | `float \| None` | `null` | `{"minimum": -2, "maximum": 2}` |
| `phase` | `float \| None` | `null` | `{"minimum": -20, "maximum": 20}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `rise` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |

<a id="experiment-drag-calibration"></a>

### `drag_calibration`

```python
drag_calibration(k: int, betas: list[float], shots: int = 400, reps: int = 1) -> ExperimentReceipt
```

Scan DRAG coefficient with two phase-sensitive sequences.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `betas` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -5, "maximum": 5}}` |
| `shots` | `int` | `400` | `{"minimum": 16, "maximum": 20000}` |
| `reps` | `int` | `1` | `{"minimum": 1, "maximum": 128}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `betas` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -5, "maximum": 5}}` |
| `shots` | `int` | `400` | `{"minimum": 16, "maximum": 20000}` |
| `reps` | `int` | `1` | `{"minimum": 1, "maximum": 128}` |

<a id="experiment-drive-phase-scan"></a>

### `drive_phase_scan`

```python
drive_phase_scan(
    source: int,
    freq: float,
    amp: float,
    duration: float,
    phases: list[float],
    shots: int = 4096,
) -> ExperimentReceipt
```

Source phase scan measuring neighbour X/Y at a user-specified carrier.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `source` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `duration` | `float` | Required | `{"minimum": 1, "maximum": 20000}` |
| `phases` | `list[float]` | Required | `{"items": {"type": "number", "minimum": -20, "maximum": 20}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `4096` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-drive-response"></a>

### `drive_response`

```python
drive_response(
    source: int,
    freqs: list[float],
    amp: float,
    duration: float,
    preparation: list[str],
    basis: list[str],
    phase: float = 0.0,
    shots: int = 1024,
) -> ExperimentReceipt
```

Observe both qubits while scanning one RF source. Change lo_freq separately to move the image at fixed RF.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `source` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `freqs` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 0.1, "maximum": 15}, "minItems": 1, "maxItems": 129}` |
| `amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `duration` | `float` | Required | `{"minimum": 1, "maximum": 20000}` |
| `phase` | `float` | `0.0` | `{"minimum": -20, "maximum": 20}` |
| `preparation` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["0", "1", "2", "+", "+i"]}, "minItems": 1, "maxItems": 2}` |
| `basis` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["X", "Y", "Z"]}, "minItems": 1, "maxItems": 2}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `source` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `freqs` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 0.1, "maximum": 15}, "minItems": 1, "maxItems": 129}` |
| `amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `duration` | `float` | Required | `{"minimum": 1, "maximum": 20000}` |
| `phase` | `float` | `0.0` | `{"minimum": -20, "maximum": 20}` |
| `preparation` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["0", "1", "2", "+", "+i"]}, "minItems": 1, "maxItems": 2}` |
| `basis` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["X", "Y", "Z"]}, "minItems": 1, "maxItems": 2}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-echo"></a>

### `echo`

```python
echo(k: int, delays: list[float], shots: int = 200) -> ExperimentReceipt
```

Scan Hahn echo delay and fit exponential coherence decay.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-flux-iq-acquire"></a>

### `flux_iq_acquire`

```python
flux_iq_acquire(
    blocks: list[float | dict],
    preparation: list[str] = ['0', '0'],
    basis: list[str] = ['Z', 'Z'],
    analysis_phases: list[float] = [0.0, 0.0],
    shots: int = 1024,
) -> ExperimentReceipt
```

Acquire one two-channel IQ experiment. Blocks are waits in ns, {"xy":[{"qubit":0,"angle":1.570796,"phase":0,"amp":0.2,"duration":40,"drag":0,"freq":5}]} (XY keys except qubit optional), {"duration":100,"pulses":[{"port":2,"amp":0.1,"duration":100,"rise":20,"freq":0,"phase":0}]}, or {"gate":"cz","repetitions":1}. Z ports 0/1=data qubits when tunable, 2=coupler. XY and Z blocks run sequentially; waits include line tails.

**Schema 1 — Scene-014, Scene-015, Scene-016**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `blocks` | `list[float \| dict]` | Required | `{"maxItems": 1024, "items": {"type": ["number", "object"], "additionalProperties": true}}` |
| `preparation` | `list[str]` | `["0", "0"]` | `{"items": {"type": "string", "enum": ["0", "1", "2", "+", "+i"]}, "minItems": 2, "maxItems": 2}` |
| `basis` | `list[str]` | `["Z", "Z"]` | `{"items": {"type": "string", "enum": ["X", "Y", "Z"]}, "minItems": 2, "maxItems": 2}` |
| `analysis_phases` | `list[float]` | `[0.0, 0.0]` | `{"items": {"type": "number", "minimum": -100, "maximum": 100}, "minItems": 2, "maxItems": 2}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-flux-iq-scan"></a>

### `flux_iq_scan`

```python
flux_iq_scan(kind: str, requests: list[dict], shots: int = 1024) -> ExperimentReceipt
```

A predefined scan costs one standard experiment, 1..441 acquisitions. Each request contains blocks, preparation, basis, analysis_phases, controls as needed. controls groups: readout={field:{"0":value,"1":value}}, flux={qubit_bias:{"0":value},coupler_bias:value,mix:3x3,inverse_tau:{"2":value},inverse_tail:{"2":value}}, native={gate_field:value}. Only the listed axes for the chosen family may vary, consistently across rows; no controls are installed by scans. qubit_spectrum/rabi/drag use XY blocks; ramsey/coherence use waits with + preparation and X/Y analysis; flux_ramsey uses Z blocks/waits; native_phase uses one CZ block and varies one gate field per scan. Allowed axes: {'readout_spectrum': {'readout': ['readout_freq']}, 'qubit_spectrum': {}, 'rabi': {}, 'iq': {}, 'ramsey': {}, 'drag': {}, 'coherence': {}, 'readout_power': {'readout': ['readout_amplitude']}, 'readout_duration': {'readout': ['readout_duration']}, 'flux_spectroscopy': {'flux': ['coupler_bias', 'qubit_bias']}, 'native_phase': {'native': ['amplitudes', 'duration', 'frequency']}, 'flux_ramsey': {'flux': ['coupler_bias', 'inverse_tail', 'inverse_tau', 'mix', 'qubit_bias']}}. Blocks are waits in ns, {"xy":[{"qubit":0,"angle":1.570796,"phase":0,"amp":0.2,"duration":40,"drag":0,"freq":5}]} (XY keys except qubit optional), {"duration":100,"pulses":[{"port":2,"amp":0.1,"duration":100,"rise":20,"freq":0,"phase":0}]}, or {"gate":"cz","repetitions":1}. Z ports 0/1=data qubits when tunable, 2=coupler. XY and Z blocks run sequentially; waits include line tails.

**Schema 1 — Scene-014, Scene-015, Scene-016**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `kind` | `str` | Required | `{"enum": ["readout_spectrum", "qubit_spectrum", "rabi", "iq", "ramsey", "drag", "coherence", "readout_power", "readout_duration", "flux_spectroscopy", "native_phase", "flux_ramsey"]}` |
| `requests` | `list[dict]` | Required | `{"minItems": 1, "maxItems": 441, "items": {"type": "object", "additionalProperties": true}}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-full-xy-scan"></a>

### `full_xy_scan`

```python
full_xy_scan(control: int, target: int, angles: list[float], shots: int = 1024) -> ExperimentReceipt
```

Prepare both qubits with X90, apply installed CNOT, sweep both XY analysis phases together; return four measured joint probabilities.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |
| `angles` | `list[float]` | Required | `{"minItems": 9, "maxItems": 257, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

<a id="experiment-gate-fringe"></a>

### `gate_fringe`

```python
gate_fringe(angles: list[float], shots: int = 4096) -> ExperimentReceipt
```

Installed CZ analysis-phase scan under both neighbour states.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `angles` | `list[float]` | Required | `{"items": {"type": "number", "minimum": -20, "maximum": 20}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `4096` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-gate-phase-scan"></a>

### `gate_phase_scan`

```python
gate_phase_scan(gate: str, n_list: list[int], shots: int = 4096) -> ExperimentReceipt
```

Repeated installed-gate X/Y scan; CNOT ++ phases or CZ conditional phase residual relative to N*pi, and measured contrast.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `gate` | `str` | Required | `{"enum": ["cnot", "cz"]}` |
| `n_list` | `list[int]` | Required | `{"items": {"type": "integer", "minimum": 0, "maximum": 16}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `4096` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-gate-quadratures"></a>

### `gate_quadratures`

```python
gate_quadratures(gate: str, qubits: list[int], shots: int = 1024) -> ExperimentReceipt
```

Measured X/Y output vectors. CNOT ++ samples both qubits together; CZ uses both neighbour states.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `gate` | `str` | Required | `{"enum": ["cnot", "cz"]}` |
| `qubits` | `list[int]` | Required | `{"items": {"type": "integer", "minimum": 0, "maximum": 1}, "minItems": 1, "maxItems": 2, "uniqueItems": true}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-gmm3-readout"></a>

### `gmm3_readout`

```python
gmm3_readout(k: int, shots: int = 2000, update_cal: bool = False) -> ExperimentReceipt
```

Prepare 0/1/2 and fit IQ clusters; optional explicit calibration update.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `shots` | `int` | `2000` | `{"minimum": 16, "maximum": 20000}` |
| `update_cal` | `bool` | `false` | — |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `shots` | `int` | `2000` | `{"minimum": 16, "maximum": 20000}` |
| `update_cal` | `bool` | `false` | — |

<a id="experiment-gmm-readout"></a>

### `gmm_readout`

```python
gmm_readout(k: int, shots: int = 2000, update_cal: bool = False) -> ExperimentReceipt
```

Prepare 0/1 and fit IQ clouds; optional explicit discriminator update.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `shots` | `int` | `2000` | `{"minimum": 16, "maximum": 20000}` |
| `update_cal` | `bool` | `false` | — |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `shots` | `int` | `2000` | `{"minimum": 16, "maximum": 20000}` |
| `update_cal` | `bool` | `false` | — |

<a id="experiment-idle-coherence-scan"></a>

### `idle_coherence_scan`

```python
idle_coherence_scan(delays: list[float], shots: int = 1024) -> ExperimentReceipt
```

Installed idle X/Y versus waiting time for each qubit and both neighbour states; eight acquisitions per point.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `delays` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 200, "maximum": 20000}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-idle-response"></a>

### `idle_response`

```python
idle_response(
    preparation: list[str],
    basis: list[str],
    delays: list[float],
    shots: int = 1024,
    installed: bool = False,
) -> ExperimentReceipt
```

Measure product-Pauli responses versus bare or installed idle; installed mode includes finite DD and hold settings.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-023, Scene-024, Scene-025, Scene-026, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `preparation` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["0", "1", "2", "+", "+i"]}, "minItems": 1, "maxItems": 2}` |
| `basis` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["X", "Y", "Z"]}, "minItems": 1, "maxItems": 2}` |
| `delays` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 0, "maximum": 200000}, "minItems": 1, "maxItems": 129}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |
| `installed` | `bool` | `false` | — |

<a id="experiment-leakage-vs-pulses"></a>

### `leakage_vs_pulses`

```python
leakage_vs_pulses(k: int, n_list: list[int], drag: float | None = None, shots: int = 1000) -> ExperimentReceipt
```

Level-2 population versus pulse count, assigned with the installed three-state readout calibration.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `n_list` | `list[int]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "integer", "minimum": 0, "maximum": 256}}` |
| `drag` | `float \| None` | `null` | `{"minimum": -5, "maximum": 5}` |
| `shots` | `int` | `1000` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `n_list` | `list[int]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "integer", "minimum": 0, "maximum": 256}}` |
| `drag` | `float \| None` | `null` | `{"minimum": -5, "maximum": 5}` |
| `shots` | `int` | `1000` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-machine-probe"></a>

### `machine_probe`

```python
machine_probe(
    preparation: list[str],
    basis: list[str],
    gate: str = 'x90',
    k: int = 0,
    edge: list[int] | None = None,
    phases: list[float] | None = None,
    gates: list[str] | None = None,
    duration: float = 1000.0,
    repetitions: int = 1,
    shots: int = 1024,
) -> ExperimentReceipt
```

Prepare product Pauli states, execute installed local/parallel/edge/idle gates, measure joint outcomes in selected bases. Includes SPAM; no author score returned.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `preparation` | `list[str]` | Required | `{"minItems": 1, "maxItems": 1, "items": {"type": "string", "enum": ["0", "1", "+", "+i"]}}` |
| `basis` | `list[str]` | Required | `{"minItems": 1, "maxItems": 1, "items": {"type": "string", "enum": ["X", "Y", "Z"]}}` |
| `gate` | `str` | `"x90"` | `{"enum": ["x90", "x180", "id", "parallel", "cnot", "cz", "zz_idle"]}` |
| `k` | `int` | `0` | `{"minimum": 0, "maximum": 0}` |
| `edge` | `list[int] \| None` | `null` | `{"minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` |
| `phases` | `list[float] \| None` | `null` | `{"minItems": 1, "maxItems": 1, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |
| `gates` | `list[str] \| None` | `null` | `{"minItems": 1, "maxItems": 1, "items": {"type": "string", "enum": ["x90", "x180"]}}` |
| `duration` | `float` | `1000.0` | `{"minimum": 1, "maximum": 20000}` |
| `repetitions` | `int` | `1` | `{"minimum": 1, "maximum": 32}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `preparation` | `list[str]` | Required | `{"minItems": 2, "maxItems": 2, "items": {"type": "string", "enum": ["0", "1", "+", "+i"]}}` |
| `basis` | `list[str]` | Required | `{"minItems": 2, "maxItems": 2, "items": {"type": "string", "enum": ["X", "Y", "Z"]}}` |
| `gate` | `str` | `"x90"` | `{"enum": ["x90", "x180", "id", "parallel", "cnot", "cz", "zz_idle"]}` |
| `k` | `int` | `0` | `{"minimum": 0, "maximum": 1}` |
| `edge` | `list[int] \| None` | `null` | `{"minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` |
| `phases` | `list[float] \| None` | `null` | `{"minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |
| `gates` | `list[str] \| None` | `null` | `{"minItems": 2, "maxItems": 2, "items": {"type": "string", "enum": ["x90", "x180"]}}` |
| `duration` | `float` | `1000.0` | `{"minimum": 1, "maximum": 20000}` |
| `repetitions` | `int` | `1` | `{"minimum": 1, "maximum": 32}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-mixer-spectrum"></a>

### `mixer_spectrum`

```python
mixer_spectrum(source: int, rf_frequency: float = 5.1, amplitude: float = 0.2, shots: int = 1024) -> ExperimentReceipt
```

Acquire mixer spectrum measurements.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `source` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `rf_frequency` | `float` | `5.1` | `{"minimum": 0.1, "maximum": 15}` |
| `amplitude` | `float` | `0.2` | `{"minimum": 0, "maximum": 1}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `source` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `rf_frequency` | `float` | `5.1` | `{"minimum": 0.1, "maximum": 15}` |
| `amplitude` | `float` | `0.2` | `{"minimum": 0, "maximum": 1}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-phase-compensation-scan"></a>

### `phase_compensation_scan`

```python
phase_compensation_scan(k: int, coefficients: list[float], repetitions: int = 3, shots: int = 1024) -> ExperimentReceipt
```

phase compensation scan

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `coefficients` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |
| `repetitions` | `int` | `3` | `{"minimum": 1, "maximum": 8}` |
| `shots` | `int` | `1024` | `{"minimum": 32, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `coefficients` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |
| `repetitions` | `int` | `3` | `{"minimum": 1, "maximum": 8}` |
| `shots` | `int` | `1024` | `{"minimum": 32, "maximum": 4096}` |

<a id="experiment-pingpong"></a>

### `pingpong`

```python
pingpong(k: int, n_list: list[int], shots: int = 200, amp: float | None = None) -> ExperimentReceipt
```

X90 followed by N X180 gates, measured versus N.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `n_list` | `list[int]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "integer", "minimum": 0, "maximum": 256}}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `amp` | `float \| None` | `null` | `{"minimum": -2, "maximum": 2}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `n_list` | `list[int]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "integer", "minimum": 0, "maximum": 256}}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `amp` | `float \| None` | `null` | `{"minimum": -2, "maximum": 2}` |

<a id="experiment-power-rabi-amplification"></a>

### `power_rabi_amplification`

```python
power_rabi_amplification(
    k: int,
    amp_factors: list[float],
    n_list: list[int],
    operation: str = 'x180',
    shots: int = 50,
) -> ExperimentReceipt
```

Ground preparation, N repeated X180 or X90 pulses, measured P1 over pulse amplitude and count. No signed-error fit or automatic calibration update.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `amp_factors` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.001, "maximum": 2}}` |
| `n_list` | `list[int]` | Required | `{"minItems": 1, "maxItems": 100, "items": {"type": "integer", "minimum": 1, "maximum": 200}}` |
| `operation` | `str` | `"x180"` | `{"enum": ["x180", "x90"]}` |
| `shots` | `int` | `50` | `{"minimum": 16, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `amp_factors` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.001, "maximum": 2}}` |
| `n_list` | `list[int]` | Required | `{"minItems": 1, "maxItems": 100, "items": {"type": "integer", "minimum": 1, "maximum": 200}}` |
| `operation` | `str` | `"x180"` | `{"enum": ["x180", "x90"]}` |
| `shots` | `int` | `50` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-pump-probe"></a>

### `pump_probe`

```python
pump_probe(
    k: int,
    pump_freq: float,
    pump_amp: float,
    pump_duration: float,
    delays: list[float],
    probe: str = 'ground',
    shots: int = 512,
) -> ExperimentReceipt
```

pump probe

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `pump_freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `pump_amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `pump_duration` | `float` | Required | `{"minimum": 0.1, "maximum": 20000}` |
| `delays` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `probe` | `str` | `"ground"` | `{"enum": ["ground", "excited", "coherence"]}` |
| `shots` | `int` | `512` | `{"minimum": 32, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `pump_freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `pump_amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `pump_duration` | `float` | Required | `{"minimum": 0.1, "maximum": 20000}` |
| `delays` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `probe` | `str` | `"ground"` | `{"enum": ["ground", "excited", "coherence"]}` |
| `shots` | `int` | `512` | `{"minimum": 32, "maximum": 4096}` |

<a id="experiment-qubit-spec"></a>

### `qubit_spec`

```python
qubit_spec(
    k: int,
    freqs: list[float],
    amp: float = 0.02,
    duration: float = 2000.0,
    shots: int = 200,
    shape: str = 'gaussian_square',
) -> ExperimentReceipt
```

Scan drive frequency; return IQ response, P1 and Lorentzian fit.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `amp` | `float` | `0.02` | `{"minimum": -2, "maximum": 2}` |
| `duration` | `float` | `2000.0` | `{"minimum": 0.01, "maximum": 200000}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `shape` | `str` | `"gaussian_square"` | `{"maxLength": 80, "enum": ["gaussian", "square", "gaussian_square"]}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `amp` | `float` | `0.02` | `{"minimum": -2, "maximum": 2}` |
| `duration` | `float` | `2000.0` | `{"minimum": 0.01, "maximum": 200000}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `shape` | `str` | `"gaussian_square"` | `{"maxLength": 80, "enum": ["gaussian", "square", "gaussian_square"]}` |

<a id="experiment-rabi-amp"></a>

### `rabi_amp`

```python
rabi_amp(k: int, amps: list[float], duration: float | None = None, shots: int = 200) -> ExperimentReceipt
```

Scan Gaussian pulse amplitude and fit oscillations.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `amps` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `duration` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `amps` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `duration` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-rabi-phase-scan"></a>

### `rabi_phase_scan`

```python
rabi_phase_scan(k: int, amps: list[float], phases: list[float], shots: int = 512) -> ExperimentReceipt
```

Amplitude Rabi at multiple command phases; fit phase-dependent effective drive gain from sampled populations.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `amps` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 0, "maximum": 2}, "minItems": 11, "maxItems": 41}` |
| `phases` | `list[float]` | Required | `{"items": {"type": "number", "minimum": -7, "maximum": 7}, "minItems": 5, "maxItems": 9}` |
| `shots` | `int` | `512` | `{"minimum": 16, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `amps` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 0, "maximum": 2}, "minItems": 11, "maxItems": 41}` |
| `phases` | `list[float]` | Required | `{"items": {"type": "number", "minimum": -7, "maximum": 7}, "minItems": 5, "maxItems": 9}` |
| `shots` | `int` | `512` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-rabi-time"></a>

### `rabi_time`

```python
rabi_time(
    k: int,
    durations: list[float],
    amp: float | None = None,
    shots: int = 200,
    rise: float = 5.0,
) -> ExperimentReceipt
```

Scan pulse duration and fit damped oscillations.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `durations` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `amp` | `float \| None` | `null` | `{"minimum": -2, "maximum": 2}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `rise` | `float` | `5.0` | `{"minimum": 0.01, "maximum": 200000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `durations` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `amp` | `float \| None` | `null` | `{"minimum": -2, "maximum": 2}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `rise` | `float` | `5.0` | `{"minimum": 0.01, "maximum": 200000}` |

<a id="experiment-ramsey"></a>

### `ramsey`

```python
ramsey(
    k: int,
    delays: list[float],
    virtual_detuning: float = 0.002,
    shots: int = 200,
    spectator: dict | None = None,
) -> ExperimentReceipt
```

Scan Ramsey delay and fit damped oscillations; the fitted frequency is unsigned.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `spectator` | `dict \| None` | `null` | `{"additionalProperties": {"type": "integer", "enum": [0, 1]}, "description": "Map qubit index strings to preparation level 0 or 1."}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `spectator` | `dict \| None` | `null` | `{"additionalProperties": {"type": "integer", "enum": [0, 1]}, "description": "Map qubit index strings to preparation level 0 or 1."}` |

<a id="experiment-ramsey-with-tones"></a>

### `ramsey_with_tones`

```python
ramsey_with_tones(
    k: int,
    delays: list[float],
    tones: list[list[float]],
    virtual_detuning: float = 0.002,
    shots: int = 200,
    spectator: dict | None = None,
    name: str = 'ramsey_tones',
    rise: float = 60.0,
    _make_figure: float = True,
) -> ExperimentReceipt
```

Run Ramsey with specified (line, frequency GHz, amplitude, phase rad) tones during delays.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `tones` | `list[list[float]]` | Required | `{"minItems": 1, "maxItems": 8, "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `spectator` | `dict \| None` | `null` | `{"additionalProperties": {"type": "integer", "enum": [0, 1]}, "description": "Map qubit index strings to preparation level 0 or 1."}` |
| `name` | `str` | `"ramsey_tones"` | `{"maxLength": 80}` |
| `rise` | `float` | `60.0` | `{"minimum": 0.01, "maximum": 200000}` |
| `_make_figure` | `float` | `true` | `{"minimum": -20, "maximum": 20}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `tones` | `list[list[float]]` | Required | `{"minItems": 1, "maxItems": 8, "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}}` |
| `virtual_detuning` | `float` | `0.002` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `spectator` | `dict \| None` | `null` | `{"additionalProperties": {"type": "integer", "enum": [0, 1]}, "description": "Map qubit index strings to preparation level 0 or 1."}` |
| `name` | `str` | `"ramsey_tones"` | `{"maxLength": 80}` |
| `rise` | `float` | `60.0` | `{"minimum": 0.01, "maximum": 200000}` |
| `_make_figure` | `float` | `true` | `{"minimum": -20, "maximum": 20}` |

<a id="experiment-readout-frequency-scan"></a>

### `readout_frequency_scan`

```python
readout_frequency_scan(k: int, freqs: list[float], shots: int = 1024) -> ExperimentReceipt
```

Prepare ground and excited states at each readout frequency and return measured IQ distances; restore the original readout frequency after acquisition.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-readout-joint-scan"></a>

### `readout_joint_scan`

```python
readout_joint_scan(
    k: int,
    amplitudes: list[float],
    frequencies: list[float],
    durations: list[float],
    shots: int = 1024,
) -> ExperimentReceipt
```

readout joint scan

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `amplitudes` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 0.01, "maximum": 1000}}` |
| `frequencies` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `durations` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 1, "maximum": 20000}}` |
| `shots` | `int` | `1024` | `{"minimum": 32, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `amplitudes` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 0.01, "maximum": 1000}}` |
| `frequencies` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `durations` | `list[float]` | Required | `{"minItems": 1, "maxItems": 101, "items": {"type": "number", "minimum": 1, "maximum": 20000}}` |
| `shots` | `int` | `1024` | `{"minimum": 32, "maximum": 4096}` |

<a id="experiment-readout-window-scan"></a>

### `readout_window_scan`

```python
readout_window_scan(k: int, durations: list[float], shots: int = 4096) -> ExperimentReceipt
```

Two prepared states versus integration time, fixed classifier; restores integration duration.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `durations` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 100, "maximum": 20000}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `4096` | `{"minimum": 16, "maximum": 4096}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `durations` | `list[float]` | Required | `{"items": {"type": "number", "minimum": 100, "maximum": 20000}, "minItems": 3, "maxItems": 41}` |
| `shots` | `int` | `4096` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-repeated-gate"></a>

### `repeated_gate`

```python
repeated_gate(k: int, n_list: list[int], gate: str = 'x90', shots: int = 200) -> ExperimentReceipt
```

N repeated X90 gates, measured versus N.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `n_list` | `list[int]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "integer", "minimum": 0, "maximum": 256}}` |
| `gate` | `str` | `"x90"` | `{"maxLength": 80, "enum": ["x90"]}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `n_list` | `list[int]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "integer", "minimum": 0, "maximum": 256}}` |
| `gate` | `str` | `"x90"` | `{"maxLength": 80, "enum": ["x90"]}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-resonator-spec"></a>

### `resonator_spec`

```python
resonator_spec(k: int, freqs: list[float], n_avg: int = 200, excited: bool = False) -> ExperimentReceipt
```

Scan readout frequency; return averaged S21 magnitude and Lorentzian fit.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `n_avg` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `excited` | `bool` | `false` | — |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `n_avg` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |
| `excited` | `bool` | `false` | — |

<a id="experiment-single-qubit-rb"></a>

### `single_qubit_rb`

```python
single_qubit_rb(k: int, lengths: list[int], sequences: int = 4, shots: int = 256, seed: int = 20261005) -> ExperimentReceipt
```

Sample single-qubit Clifford sequences and their inverses; estimate RB decay from measured survival including SPAM.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `lengths` | `list[int]` | Required | `{"minItems": 1, "maxItems": 21, "items": {"type": "integer", "minimum": 1, "maximum": 64}}` |
| `sequences` | `int` | `4` | `{"minimum": 1, "maximum": 20}` |
| `shots` | `int` | `256` | `{"minimum": 16, "maximum": 4096}` |
| `seed` | `int` | `20261005` | `{"minimum": 0}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `lengths` | `list[int]` | Required | `{"minItems": 1, "maxItems": 21, "items": {"type": "integer", "minimum": 1, "maximum": 64}}` |
| `sequences` | `int` | `4` | `{"minimum": 1, "maximum": 20}` |
| `shots` | `int` | `256` | `{"minimum": 16, "maximum": 4096}` |
| `seed` | `int` | `20261005` | `{"minimum": 0}` |

<a id="experiment-sizzle-zz-scan"></a>

### `sizzle_zz_scan`

```python
sizzle_zz_scan(
    i: int,
    j: int,
    tone_freq: float,
    amps: list[float],
    delays: list[float],
    phase_j: float = 0.0,
    amp_ratio: float = 1.0,
    virtual_detuning: float = 0.003,
    shots: int = 200,
) -> ExperimentReceipt
```

Measure conditional Ramsey ZZ versus simultaneous tone amplitude.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `i` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `j` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `tone_freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `amps` | `list[float]` | Required | `{"minItems": 2, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `phase_j` | `float` | `0.0` | `{"minimum": -20, "maximum": 20}` |
| `amp_ratio` | `float` | `1.0` | `{"minimum": -2, "maximum": 2}` |
| `virtual_detuning` | `float` | `0.003` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `i` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `j` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `tone_freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `amps` | `list[float]` | Required | `{"minItems": 2, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `phase_j` | `float` | `0.0` | `{"minimum": -20, "maximum": 20}` |
| `amp_ratio` | `float` | `1.0` | `{"minimum": -2, "maximum": 2}` |
| `virtual_detuning` | `float` | `0.003` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-spec-vs-power"></a>

### `spec_vs_power`

```python
spec_vs_power(
    k: int,
    freqs: list[float],
    amps: list[float],
    duration: float = 2000.0,
    shots: int = 150,
    shape: str = 'gaussian',
) -> ExperimentReceipt
```

Repeat spectroscopy over drive amplitudes and return per-power traces.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `amps` | `list[float]` | Required | `{"minItems": 2, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `duration` | `float` | `2000.0` | `{"minimum": 0.01, "maximum": 200000}` |
| `shots` | `int` | `150` | `{"minimum": 16, "maximum": 20000}` |
| `shape` | `str` | `"gaussian"` | `{"maxLength": 80, "enum": ["gaussian", "square", "gaussian_square"]}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `freqs` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0.1, "maximum": 15}}` |
| `amps` | `list[float]` | Required | `{"minItems": 2, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `duration` | `float` | `2000.0` | `{"minimum": 0.01, "maximum": 200000}` |
| `shots` | `int` | `150` | `{"minimum": 16, "maximum": 20000}` |
| `shape` | `str` | `"gaussian"` | `{"maxLength": 80, "enum": ["gaussian", "square", "gaussian_square"]}` |

<a id="experiment-spectator-echo-timing"></a>

### `spectator_echo_timing`

```python
spectator_echo_timing(
    probe: int,
    spectator: int,
    offsets: list[float],
    duration: float = 400.0,
    shots: int = 1024,
) -> ExperimentReceipt
```

Ramsey on one qubit with a finite spectator X180 at scanned offsets; both spectator inputs and X/Y quadratures. Differential phase distinguishes common detuning from relative timing.

**Schema 1 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `probe` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `spectator` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `offsets` | `list[float]` | Required | `{"items": {"type": "number", "minimum": -1000, "maximum": 1000}, "minItems": 3, "maxItems": 41}` |
| `duration` | `float` | `400.0` | `{"minimum": 100, "maximum": 10000}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |

<a id="experiment-standard-scan"></a>

### `standard_scan`

```python
standard_scan(kind: str, index: int | list[int], rows: list[dict], shots: int = 1024) -> ExperimentReceipt
```

One predeclared scan or tomography. No adaptive updates inside; per-row points/fits and all raw IQ retained. Same 441 acquisition cap as other experiments. Kinds: relaxation_series, resonator_power_map, rabi_chevron, readout_power, readout_duration, ramsey_quadratures, tone_quadratures, conditional_phase_scan, cnot_truth_scan, cr_cancel_map, readout_weights, gate_frames, state_tomography, mixer_lo_map, mixer_image_map

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `kind` | `str` | Required | `{"enum": ["relaxation_series", "resonator_power_map", "rabi_chevron", "readout_power", "readout_duration", "ramsey_quadratures", "tone_quadratures", "conditional_phase_scan", "cnot_truth_scan", "cr_cancel_map", "readout_weights", "gate_frames", "state_tomography", "mixer_lo_map", "mixer_image_map"]}` |
| `index` | `int \| list[int]` | Required | `{"minimum": 0, "maximum": 0, "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` |
| `rows` | `list[dict]` | Required | `{"minItems": 1, "maxItems": 441, "items": {"type": "object", "additionalProperties": false, "required": ["parameters"], "properties": {"parameters": {"type": "object", "additionalProperties": true}, "controls": {"type": "object", "additionalProperties": true}}}}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `kind` | `str` | Required | `{"enum": ["relaxation_series", "resonator_power_map", "rabi_chevron", "readout_power", "readout_duration", "ramsey_quadratures", "tone_quadratures", "conditional_phase_scan", "cnot_truth_scan", "cr_cancel_map", "readout_weights", "gate_frames", "state_tomography", "mixer_lo_map", "mixer_image_map"]}` |
| `index` | `int \| list[int]` | Required | `{"minimum": 0, "maximum": 1, "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` |
| `rows` | `list[dict]` | Required | `{"minItems": 1, "maxItems": 441, "items": {"type": "object", "additionalProperties": false, "required": ["parameters"], "properties": {"parameters": {"type": "object", "additionalProperties": true}, "controls": {"type": "object", "additionalProperties": true}}}}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-stark-shift-scan"></a>

### `stark_shift_scan`

```python
stark_shift_scan(
    k: int,
    tone_freq: float,
    amps: list[float],
    delays: list[float],
    virtual_detuning: float = 0.003,
    shots: int = 200,
) -> ExperimentReceipt
```

Measure differential Ramsey frequency shift relative to the first supplied amplitude.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `tone_freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `amps` | `list[float]` | Required | `{"minItems": 2, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.003` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `tone_freq` | `float` | Required | `{"minimum": 0.1, "maximum": 15}` |
| `amps` | `list[float]` | Required | `{"minItems": 2, "maxItems": 257, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `virtual_detuning` | `float` | `0.003` | `{"minimum": -0.1, "maximum": 0.1}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-state-readout"></a>

### `state_readout`

```python
state_readout(preparation: list[str], shots: int = 1024, time_resolved: bool = False) -> ExperimentReceipt
```

Prepare a product state and acquire raw IQ, optionally time-bin IQ; includes preparation errors.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-023, Scene-024, Scene-025, Scene-026, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `preparation` | `list[str]` | Required | `{"items": {"type": "string", "enum": ["0", "1", "2", "+", "+i"]}, "minItems": 1, "maxItems": 2}` |
| `shots` | `int` | `1024` | `{"minimum": 16, "maximum": 4096}` |
| `time_resolved` | `bool` | `false` | — |

<a id="experiment-t1"></a>

### `t1`

```python
t1(k: int, delays: list[float], shots: int = 200) -> ExperimentReceipt
```

Scan relaxation delay and fit exponential T1.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `k` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `delays` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": 0, "maximum": 200000}}` |
| `shots` | `int` | `200` | `{"minimum": 16, "maximum": 20000}` |

<a id="experiment-xy-plane-cnot-calibration"></a>

### `xy_plane_cnot_calibration`

```python
xy_plane_cnot_calibration(
    control: int,
    target: int,
    cr_amp: float,
    cr_duration: float,
    angles: list[float],
    cr_phase: float | None = None,
    shots: int = 300,
    rise: float | None = None,
) -> ExperimentReceipt
```

Scan standard target XY analysis axes after CR; fit a planar rotation axis modulo pi assuming approximately opposite conditional rotations.

**Schema 1 — Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 0}` |
| `cr_amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `cr_duration` | `float` | Required | `{"minimum": 0.01, "maximum": 200000}` |
| `angles` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |
| `cr_phase` | `float \| None` | `null` | `{"minimum": -20, "maximum": 20}` |
| `shots` | `int` | `300` | `{"minimum": 16, "maximum": 20000}` |
| `rise` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |

**Schema 2 — Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026**

| Parameter | Type | Required / default | Constraints |
| --- | --- | --- | --- |
| `control` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `target` | `int` | Required | `{"minimum": 0, "maximum": 1}` |
| `cr_amp` | `float` | Required | `{"minimum": -2, "maximum": 2}` |
| `cr_duration` | `float` | Required | `{"minimum": 0.01, "maximum": 200000}` |
| `angles` | `list[float]` | Required | `{"minItems": 5, "maxItems": 257, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |
| `cr_phase` | `float \| None` | `null` | `{"minimum": -20, "maximum": 20}` |
| `shots` | `int` | `300` | `{"minimum": 16, "maximum": 20000}` |
| `rise` | `float \| None` | `null` | `{"minimum": 0.01, "maximum": 200000}` |

## Control updates

```python
update_calibration(updates=[{"field": FIELD, "index": INDEX, "value": VALUE}])
```

Use 1–64 updates per call. JSON state keys are strings such as `"0"` or `"(0, 1)"`; update indices are integers or `[control, target]`. For Tune, preserve all scalars outside `objective.parameters_to_fix`, including the unchanged elements of an array or object value. A field can be listed in a shared control schema even when a device has no valid pair index for it.

Each entry below gives the field's exact index and value schema for its scene group. Empty calibration dictionaries mean that no value is explicitly installed for that field. Use the field description and the connected device's control constraints to interpret defaults.

**Control constraints — Scene-001, Scene-002, Scene-003, Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-023, Scene-024, Scene-025, Scene-026, Scene-027**

- All values must be finite. Updates in one call are atomic.
- Pair fields require a directed physical edge. Status dictionary keys are strings; update indices are integer or [control,target].
- Stored CR and siZZle duration >= 2*rise. ECR additionally requires total cr_duration >= 4*cr_rise.
- Compiled CNOT additionally requires each spectator pi_duration <= cr_duration/2 for scheduled echo pulses.
- Changing duration does not automatically rescale amplitude. Hardware laws remain fixed; flux_bias changes their commanded operating point.
- DD uses finite pulses inside the requested idle window; CPMG count must be even. Readout weights cannot be all zero.

**Control constraints — Scene-014, Scene-015, Scene-016**

- Updates are atomic; only calibration controls change.
- Flux pulses must remain within physical flux ranges after mixing/filtering.
- Scans restore controls afterwards; explicitly update calibration to install a result.
- Cold start has empty controls. Set approximate drive/readout frequencies before preparing excited states.

### `adc_gain`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Positive linear receiver gain correction (not dB).

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.05, "maximum": 20.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Positive linear receiver gain correction (not dB).

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.05, "maximum": 20.0}` |

### `adc_offset_i`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Input-referred additive I offset correction for the linear receiver.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Input-referred additive I offset correction for the linear receiver.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

### `adc_offset_q`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Input-referred additive Q offset correction for the linear receiver.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Input-referred additive Q offset correction for the linear receiver.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

### `ampm_compensation`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Memoryless source phase predistortion c: z -> z exp(i c |z|^2), before physical AM-PM distortion; rad/amplitude squared. Applies to the summed source waveform, including DRAG.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -20.0, "maximum": 20.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Memoryless source phase predistortion c: z -> z exp(i c |z|^2), before physical AM-PM distortion; rad/amplitude squared. Applies to the summed source waveform, including DRAG.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -20.0, "maximum": 20.0}` |

### `anharm`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Believed f12-f01 (GHz); used by DRAG and f12 preparation.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -1, "maximum": -0.001}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Believed f12-f01 (GHz); used by DRAG and f12 preparation.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -1, "maximum": -0.001}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

Estimated f12-f01 for DRAG, GHz.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": -2.0, "maximum": -0.001}` |

### `channel_delay`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Signed source timing correction in ns. Advanced/delayed tails are retained and charged.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -100.0, "maximum": 100.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Signed source timing correction in ns. Advanced/delayed tails are retained and charged.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -100.0, "maximum": 100.0}` |

### `cnot_frame_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Explicit virtual-Z frame increment after each compiled CNOT (radians), added to active-pair post_gate_phase and fixed control -pi/2. It is applied to all subsequent probes.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Explicit virtual-Z frame increment after each compiled CNOT (radians), added to active-pair post_gate_phase and fixed control -pi/2. It is applied to all subsequent probes.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `coupler_flux_bias`

**Scenes:** Scene-014, Scene-015, Scene-016.

Coupler DC flux, Phi0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0]}` | `{"type": "number", "minimum": -0.45, "maximum": 0.45}` |

### `cr_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

CR source amplitude.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -2, "maximum": 2}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

CR source amplitude.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -2, "maximum": 2}` |

### `cr_cancel`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Complex target cancellation amplitude relative to CR phase.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "object", "required": ["real", "imag"], "additionalProperties": false, "properties": {"real": {"type": "number", "minimum": -2, "maximum": 2}, "imag": {"type": "number", "minimum": -2, "maximum": 2}}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Complex target cancellation amplitude relative to CR phase.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "object", "required": ["real", "imag"], "additionalProperties": false, "properties": {"real": {"type": "number", "minimum": -2, "maximum": 2}, "imag": {"type": "number", "minimum": -2, "maximum": 2}}}` |

### `cr_drag`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Derivative coefficient for CR waveform.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -5, "maximum": 5}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Derivative coefficient for CR waveform.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -5, "maximum": 5}` |

### `cr_duration`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Total ECR driven duration (ns), excluding finite echo/local pulses.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 1, "maximum": 20000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Total ECR driven duration (ns), excluding finite echo/local pulses.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 1, "maximum": 20000}` |

### `cr_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

CR source phase (radians).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

CR source phase (radians).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `cr_rise`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Rise duration of each Gaussian-square CR pulse (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 0.1, "maximum": 10000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Rise duration of each Gaussian-square CR pulse (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 0.1, "maximum": 10000}` |

### `cr_rotary`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Additional complex target tone in echoed CR, relative to CR phase.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "object", "required": ["real", "imag"], "additionalProperties": false, "properties": {"real": {"type": "number", "minimum": -2, "maximum": 2}, "imag": {"type": "number", "minimum": -2, "maximum": 2}}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Additional complex target tone in echoed CR, relative to CR phase.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "object", "required": ["real", "imag"], "additionalProperties": false, "properties": {"real": {"type": "number", "minimum": -2, "maximum": 2}, "imag": {"type": "number", "minimum": -2, "maximum": 2}}}` |

### `dd_min_duration`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Apply DD only at or above this idle duration in ns (default 200). Pulses must fit without clipping.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.0, "maximum": 20000.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Apply DD only at or above this idle duration in ns (default 200). Pulses must fit without clipping.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.0, "maximum": 20000.0}` |

### `dd_offset`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Per-qubit DD center offset as a fraction of pulse spacing; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -0.45, "maximum": 0.45}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Per-qubit DD center offset as a fraction of pulse spacing; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -0.45, "maximum": 0.45}` |

### `dd_pulses`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Even CPMG pulse count (default 2). XY4/XY8 use 4/8 pulses.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "integer", "minimum": 2, "maximum": 32}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Even CPMG pulse count (default 2). XY4/XY8 use 4/8 pulses.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "integer", "minimum": 2, "maximum": 32}` |

### `dd_sequence`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Installed finite-pulse idle program: off/cpmg/xy4/xy8. Preserves the requested total idle window.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "string", "enum": ["off", "cpmg", "xy4", "xy8"]}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Installed finite-pulse idle program: off/cpmg/xy4/xy8. Preserves the requested total idle window.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "string", "enum": ["off", "cpmg", "xy4", "xy8"]}` |

### `drag`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Gaussian derivative quadrature coefficient; uses installed anharmonicity.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -5, "maximum": 5}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Gaussian derivative quadrature coefficient; uses installed anharmonicity.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -5, "maximum": 5}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

drag

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": -5.0, "maximum": 5.0}` |

### `dressing_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Continuous microwave dressing source amplitude (default zero). Recalibrate drive carrier and pulses after changing the dressed operating point.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.0, "maximum": 2.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Continuous microwave dressing source amplitude (default zero). Recalibrate drive carrier and pulses after changing the dressed operating point.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.0, "maximum": 2.0}` |

### `dressing_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Continuous off-resonant microwave dressing carrier, GHz (default 5.2). Physically shifts the qubit through the AC Stark effect. Three-level backends use a dispersive approximation; the full-cosine backend propagates the continuous tone explicitly.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.1, "maximum": 15.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Continuous off-resonant microwave dressing carrier, GHz (default 5.2). Physically shifts the qubit through the AC Stark effect. Three-level backends use a dispersive approximation; the full-cosine backend propagates the continuous tone explicitly.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.1, "maximum": 15.0}` |

### `drive_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Installed f01 drive carrier (GHz); experiments may suggest but do not install it.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Installed f01 drive carrier (GHz); experiments may suggest but do not install it.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

Microwave carrier GHz; independent of physical flux bias.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": 0.1, "maximum": 15.0}` |

### `ecr_balance`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

First/second CR amplitudes scale by 1+balance and 1-balance.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -0.99, "maximum": 0.99}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

First/second CR amplitudes scale by 1+balance and 1-balance.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -0.99, "maximum": 0.99}` |

### `ecr_phase_skew`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Extra phase on second CR half only (radians).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Extra phase on second CR half only (radians).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `ecr_walsh`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

1 splits CR into four driven quarters and uses distinct finite spectator echo patterns; 0 retains the two-half sequence.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

1 splits CR into four driven quarters and uses distinct finite spectator echo patterns; 0 retains the two-half sequence.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

### `flux_bias`

**Scenes:** Scene-015, Scene-016.

Independent data-qubit DC flux, Phi0. Recalibrate microwave frequency after changing it.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": -0.45, "maximum": 0.45}` |

### `flux_inverse_tail`

**Scenes:** Scene-014.

Fraction of step response in the inverted exponential tail.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [2]}` | `{"type": "number", "minimum": 1e-06, "maximum": 1}` |

**Scenes:** Scene-015, Scene-016.

Fraction of step response in the inverted exponential tail.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1, 2]}` | `{"type": "number", "minimum": 1e-06, "maximum": 1}` |

### `flux_inverse_tau`

**Scenes:** Scene-014.

Inverse flux-filter time constant per Z port, ns; zero disables.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [2]}` | `{"type": "number", "minimum": 0, "maximum": 20000}` |

**Scenes:** Scene-015, Scene-016.

Inverse flux-filter time constant per Z port, ns; zero disables.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1, 2]}` | `{"type": "number", "minimum": 0, "maximum": 20000}` |

### `flux_mix`

**Scenes:** Scene-014, Scene-015, Scene-016.

3x3 digital flux crosstalk compensation. Only installed Z output ports may be used.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0]}` | `{"type": "array", "items": {"type": "array", "items": {"type": "number", "minimum": -10, "maximum": 10}, "minItems": 3, "maxItems": 3}, "minItems": 3, "maxItems": 3}` |

### `hold_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |

### `hold_enabled`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

0: bare idle. 1: use the installed siZZle cancellation block for each complete 1000 ns of pair idle; remaining time is bare. The same physical pulse and frame increment are used by probes and scoring. Applies to pair devices only.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

0: bare idle. 1: use the installed siZZle cancellation block for each complete 1000 ns of pair idle; remaining time is bare. The same physical pulse and frame increment are used by probes and scoring. Applies to pair devices only.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

### `hold_frame`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Installed local logical-frame increments for one 1000 ns cancellation block, radians; never fitted by the scorer.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Installed local logical-frame increments for one 1000 ns cancellation block, radians; never fitted by the scorer.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

### `hold_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

### `hold_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `hold_rise`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 0.1, "maximum": 500}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Independent siZZle idle workpoint

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 0.1, "maximum": 500}` |

### `hold_smooth`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Independent idle-cancellation ramp: 1 normalizes endpoints to zero; 0 retains the original truncated envelope. Does not inherit CZ settings.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Independent idle-cancellation ramp: 1 normalizes endpoints to zero; 0 retains the original truncated envelope. Does not inherit CZ settings.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

### `idle_active`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-023, Scene-024, Scene-025, Scene-026, Scene-027.

Key 0: enable the installed whole-device 1000 ns cancellation block. All tones are physical; the remainder is a bare wait.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

### `idle_frame_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Installed frame increment for this qubit after a physical whole-device idle block, radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Installed frame increment for this qubit after a physical whole-device idle block, radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `idle_tone_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Off-resonant idle tone amplitude on this logical input.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0, "maximum": 2}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Off-resonant idle tone amplitude on this logical input.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0, "maximum": 2}` |

### `idle_tone_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Off-resonant idle tone carrier on this logical input, GHz.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Off-resonant idle tone carrier on this logical input, GHz.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

### `idle_tone_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Laboratory phase of this idle tone; does not follow computational virtual frames, radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Laboratory phase of this idle tone; does not follow computational virtual frames, radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `idle_tone_rise`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Rise/fall duration of this idle tone in the 1000 ns block, ns.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.1, "maximum": 500}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Rise/fall duration of this idle tone in the 1000 ns block, ns.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.1, "maximum": 500}` |

### `idle_tone_smooth`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

1 normalizes truncated Gaussian ramps to zero amplitude at the endpoints; 0 retains the original truncated envelope.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

1 normalizes truncated Gaussian ramps to zero amplitude at the endpoints; 0 retains the original truncated envelope.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

### `iq_angle`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Assign 1 when real(IQ*exp(-i*angle)) > iq_threshold; radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Assign 1 when real(IQ*exp(-i*angle)) > iq_threshold; radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

Classifier angle rad; assigned 1 when Re(IQ exp(-i angle)) > iq_threshold.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": -20.0, "maximum": 20.0}` |

### `iq_means3`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Ordered complex cluster means for levels 0,1,2.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "required": ["real", "imag"], "additionalProperties": false, "properties": {"real": {"type": "number", "minimum": -2000, "maximum": 2000}, "imag": {"type": "number", "minimum": -2000, "maximum": 2000}}}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Ordered complex cluster means for levels 0,1,2.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "required": ["real", "imag"], "additionalProperties": false, "properties": {"real": {"type": "number", "minimum": -2000, "maximum": 2000}, "imag": {"type": "number", "minimum": -2000, "maximum": 2000}}}}` |

### `iq_threshold`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Threshold on rotated IQ.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -2000, "maximum": 2000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Threshold on rotated IQ.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -2000, "maximum": 2000}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

iq threshold

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": -2000.0, "maximum": 2000.0}` |

### `lo_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Physical source LO in GHz (default 5.0). Commanded RF is unchanged; signed IF=RF-LO and image=2*LO-RF.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.1, "maximum": 15.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Physical source LO in GHz (default 5.0). Commanded RF is unchanged; signed IF=RF-LO and image=2*LO-RF.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.1, "maximum": 15.0}` |

### `mixer_dc_i`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Physical-source I DC offset in command amplitude units; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Physical-source I DC offset in command amplitude units; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

### `mixer_dc_q`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Physical-source Q DC offset in command amplitude units; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Physical-source Q DC offset in command amplitude units; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -2.0, "maximum": 2.0}` |

### `mixer_gain`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Digital Q/I gain before the physical IQ mixer; default 1.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.2, "maximum": 5.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Digital Q/I gain before the physical IQ mixer; default 1.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.2, "maximum": 5.0}` |

### `mixer_skew`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Digital quadrature skew in radians; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": -1.4, "maximum": 1.4}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Digital quadrature skew in radians; default 0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": -1.4, "maximum": 1.4}` |

### `native_flux_gate`

**Scenes:** Scene-014, Scene-015, Scene-016.

Installed CZ: three port amplitudes/rise fractions, carrier GHz, duration/settle ns and two logical frame corrections rad. Fixed-frequency data qubits require amplitudes[0:2]=0. Baseband frequency=0; parametric frequency>0.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0]}` | `{"type": "object", "additionalProperties": false, "properties": {"mode": {"type": "string", "enum": ["baseband_cz", "parametric_cz"]}, "duration": {"type": "number", "minimum": 1, "maximum": 20000}, "amplitudes": {"type": "array", "items": {"type": "number", "minimum": -0.45, "maximum": 0.45}, "minItems": 3, "maxItems": 3}, "rise_fractions": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 0.5}, "minItems": 3, "maxItems": 3}, "frequency": {"type": "number", "minimum": 0, "maximum": 5}, "phase": {"type": "number", "minimum": -100, "maximum": 100}, "settle": {"type": "number", "minimum": 0, "maximum": 20000}, "frames": {"type": "array", "items": {"type": "number", "minimum": -100, "maximum": 100}, "minItems": 2, "maxItems": 2}}, "required": ["mode", "duration", "amplitudes", "rise_fractions", "frequency", "phase", "settle", "frames"]}` |

### `pi_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Gaussian X180 amplitude; X90 uses half this amplitude at the same duration.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.0001, "maximum": 2}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Gaussian X180 amplitude; X90 uses half this amplitude at the same duration.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.0001, "maximum": 2}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

X180 Gaussian amplitude; X90 has half amplitude at the same pi_duration.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": 0.0, "maximum": 2.0}` |

### `pi_duration`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Total Gaussian gate duration (ns); changing it does not rescale pi_amp.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 1, "maximum": 2000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Total Gaussian gate duration (ns); changing it does not rescale pi_amp.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 1, "maximum": 2000}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

Gaussian gate duration ns; does not rescale pi_amp.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": 1.0, "maximum": 2000.0}` |

### `post_gate_network_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

One installed logical-frame increment per qubit for this edge gate; replaces legacy free frame corrections, never fitted by scorer.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

One installed logical-frame increment per qubit for this edge gate; replaces legacy free frame corrections, never fitted by scorer.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

### `post_gate_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Two installed local phase corrections, ordered control,target; compiled CNOT adds these explicit logical frame increments.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Two installed local phase corrections, ordered control,target; compiled CNOT adds these explicit logical frame increments.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

### `readout_amplitude`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Readout voltage amplitude (default 1); incident power is proportional to its square. Refit the IQ classifier after changes.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0, "maximum": 1000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Readout voltage amplitude (default 1); incident power is proportional to its square. Refit the IQ classifier after changes.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0, "maximum": 1000}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

Readout voltage; power is proportional to its square.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": 0.0, "maximum": 1000.0}` |

### `readout_duration`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Integration time (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 1, "maximum": 20000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Integration time (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 1, "maximum": 20000}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

IQ integration window ns.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": 1.0, "maximum": 20000.0}` |

### `readout_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Readout carrier (GHz).

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Readout carrier (GHz).

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-014, Scene-015, Scene-016.

readout freq

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "enum": [0, 1]}` | `{"type": "number", "minimum": 0.1, "maximum": 15.0}` |

### `readout_time_of_flight`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

ADC integration-window start relative to emitted readout, ns; distinct from drive channel_delay.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "number", "minimum": 0.0, "maximum": 2000.0}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

ADC integration-window start relative to emitted readout, ns; distinct from drive channel_delay.

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "number", "minimum": 0.0, "maximum": 2000.0}` |

### `sizzle_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Two source amplitudes, ordered pair.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Two source amplitudes, ordered pair.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}` |

### `sizzle_duration`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

siZZle pulse duration (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 1, "maximum": 20000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

siZZle pulse duration (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 1, "maximum": 20000}` |

### `sizzle_freq`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Shared off-resonant carrier (GHz).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Shared off-resonant carrier (GHz).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 0.1, "maximum": 15}` |

### `sizzle_network_amp`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

One physical CZ tone amplitude per logical input, all at sizzle_freq for this edge. Used instead of the two-tone template when installed.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "number", "minimum": 0, "maximum": 2}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

One physical CZ tone amplitude per logical input, all at sizzle_freq for this edge. Used instead of the two-tone template when installed.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": 0, "maximum": 2}}` |

### `sizzle_network_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

One laboratory CZ tone phase per logical input, radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

One laboratory CZ tone phase per logical input, radians.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -20, "maximum": 20}}` |

### `sizzle_phase`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Relative phase of second siZZle tone (radians).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Relative phase of second siZZle tone (radians).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": -20, "maximum": 20}` |

### `sizzle_rise`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

siZZle ramp duration (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "number", "minimum": 0.1, "maximum": 10000}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

siZZle ramp duration (ns).

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "number", "minimum": 0.1, "maximum": 10000}` |

### `sizzle_smooth`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

1 uses zero-endpoint normalized Gaussian ramps for this edge CZ; 0 uses the original truncated envelope.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 0}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

1 uses zero-endpoint normalized Gaussian ramps for this edge CZ; 0 uses the original truncated envelope.

| Index schema | Value schema |
| --- | --- |
| `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 1}}` | `{"type": "integer", "minimum": 0, "maximum": 1}` |

### `source_mix`

**Scenes:** Scene-001, Scene-002, Scene-003, Scene-013, Scene-017, Scene-018, Scene-019, Scene-020, Scene-021, Scene-022, Scene-027.

Source mixing: input qubit to physical source complex weights [real,imag]

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 0}` | `{"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}}` |

**Scenes:** Scene-004, Scene-005, Scene-006, Scene-007, Scene-008, Scene-009, Scene-010, Scene-011, Scene-012, Scene-023, Scene-024, Scene-025, Scene-026.

Source mixing: input qubit to physical source complex weights [real,imag]

| Index schema | Value schema |
| --- | --- |
| `{"type": "integer", "minimum": 0, "maximum": 1}` | `{"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}}` |

## Observation data and artifacts

| Observation | Where to read it |
| --- | --- |
| Current installed controls and budget | `status().state` |
| Initial acceptance metrics and task objective | `status().state.objective` |
| Fit summaries and experiment metadata | Experiment receipt: `fit`, `meta` |
| Full fit, including the fitted curve | `check_raw_data(request_id, section="fit")` |
| Scan axes, measured values and histograms | `check_raw_data(request_id, section="points", array=NAME, start=0, stop=100)` |
| Raw-array names, dimensions and acquisition metadata | `check_raw_data(request_id, section="manifest")` |
| Raw IQ, time-resolved IQ or recorded S21 arrays | `get_data(request_id, array=NAME, start=0, stop=2000)` |
| Experiment figure | `client.download(receipt["figures"][i]["artifact"])` → PNG bytes |
| Complete recorded response | `check_raw_data(request_id, section="response")` or download `response_artifact` |
| Complete raw arrays | Download `raw_data.artifact` → NPZ bytes |
| Final acceptance and score after the episode | Run directory: `final.json`, `result.json`, `calibration.json` |

Calls return JSON objects with `status="success"` or `status="failed"`. Failure responses include `error` and/or `error_type`; additional state or cost information may be included. Successful experiment execution and submission receipts describe the completed operation; final acceptance is determined after the episode.

The benchmark interface is independent of the calibrator. The optional Blueprint adapter registers these actions and experiments as `qcal_NAME` functions, accessible through Blueprint's native `run_experiment` tool. This prefix belongs to that adapter; the HTTP action and experiment names above are the shared interface.

Source definitions: [`client.py`](client.py), [`backend/benchmark/catalog.py`](backend/benchmark/catalog.py), [`backend/probes.py`](backend/probes.py), [`backend/benchmark/controls.py`](backend/benchmark/controls.py), and [`backend/flux_benchmark_session.py`](backend/flux_benchmark_session.py).
