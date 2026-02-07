# Guitar Tuner

Quick &amp; dirty microphone-based, vibe coded guitar tuner

- **GUI**: `tkinter`
- **Pitch detection**: autocorrelation + parabolic peak refinement
- **Display**: a semi-circle “LED” meter showing how flat/sharp you are (±50 cents), plus the current note and signed cents offset
- **Behavior**: starts listening automatically on launch; close the window to stop

## Requirements

- Python **3.10+**
- A working microphone input device

## Install dependencies

```bash
python3 -m pip install numpy sounddevice
```

## Run

```bash
python3 tuner.py
```

## Notes / troubleshooting

- **macOS**: you may need to grant microphone permission to your Terminal/Python.
- **Windows**: check *Microphone privacy* settings if the input stream won’t start.
