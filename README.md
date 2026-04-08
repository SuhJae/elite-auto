# Elite Auto

Elite Auto is a Windows-first Python scaffold for a narrow, modular Elite Dangerous control loop. This pass focuses on maintainable architecture, local file-backed state reading, DirectX-friendly adapters, and safe smoke-testable action/routine wiring. It does not implement autopilot logic.

## Requirements

- Windows
- Python 3.11+
- Elite Dangerous local files available under `Saved Games/Frontier Developments/Elite Dangerous`

## Setup

1. Install Python 3.11 if it is not already present.
2. Create or recreate the virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Copy `config.example.json` to a local config file and update paths or control bindings as needed.

## Architecture

- `app/domain`
  Shared protocols and dataclasses for state, events, results, and runtime context.
- `app/adapters`
  Thin wrappers around `pydirectinput`, `dxcam`, `watchdog`, and `opencv-python`.
- `app/state`
  File-backed readers for `Status.json`, `Journal*.log`, `Cargo.json`, `Market.json`, and Elite `.binds`.
- `app/actions`
  Small polling actions that depend only on interfaces and return `Result`.
- `app/routines`
  Composable routines built from action classes.
- `app/scenarios`
  Safe smoke helpers using fake readers and fake state.
- `app/tests`
  `unittest` coverage for parser and reader behavior.

## CLI Usage

Print the current state snapshot:

```powershell
.\.venv\Scripts\python.exe run.py --status
```

Run a single action:

```powershell
.\.venv\Scripts\python.exe run.py --action wait_until_undocked
```

Run the simple starport buy block with commodity parameters:

```powershell
.\.venv\Scripts\python.exe run.py --action buy_from_starport --commodity "Meta-Alloys"
```

Run the starport buy file directly:

```powershell
.\.venv\Scripts\python.exe app\actions\starport_buy.py
```

Run the sample routine:

```powershell
.\.venv\Scripts\python.exe run.py --routine sample_departure
```

Use a custom config file:

```powershell
.\.venv\Scripts\python.exe run.py --config .\config.example.json --status
```

Print the parsed market snapshot for the most recently docked station:

```powershell
.\.venv\Scripts\python.exe run.py --market
```

Print the simulated Buy screen from current market data:

```powershell
.\.venv\Scripts\python.exe run.py --buy-screen
```

## Implemented In This Pass

- Typed domain protocols for capture, input, ship control, state reading, event streaming, and vision.
- Shared `Context`, `ShipState`, `JournalEvent`, `TemplateMatch`, and `Result` types.
- `Status.json` parsing with flag mapping for docked, mass lock, and supercruise.
- Journal tailing for appended lines in the newest `Journal*.log`.
- Minimal `Cargo.json` reader and a typed `Market.json` reader with journal-based station matching.
- `pydirectinput` ship control adapter with centralized key mapping.
- Automatic `.binds` discovery from the active Elite preset, with fallback to configured defaults for unmapped controls.
- `dxcam` capture wrapper and `watchdog` file watcher service.
- OpenCV template matching utility and debug snapshot support.
- Three example wait actions and one sample routine.
- Simple `buy_from_starport` action block in its own file for reuse in routines.
- Parser tests and fake-context smoke helpers.

## TODO / Not Implemented Yet

- Modifier-aware and broader `.binds` coverage beyond the current keyboard-first action set.
- OCR heuristics and HUD-specific vision logic.
- Real docking, trading, and navigation control logic.
- A large FSM or autonomous route planner.
- Robust live game synchronization, retries, and telemetry beyond the current scaffold.

## Testing

Run the parser and tailer tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover app/tests
```

For local smoke helpers without touching the live game, inspect or call functions in `app/scenarios/smoke_tests.py`.
