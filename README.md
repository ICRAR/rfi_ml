# pyvex
Python bindings to the CSIRO vex file parser

### Installation

```bash
git clone https://github.com/ICRAR/pyvex.git
cd pyvex/pyvex
python setup.py install
```

### Usage

```python
from pyvex import Vex
vex_file = Vex("sample.vex")

print(vex_file.directory)
print(vex_file.polarisations)
print(vex_file.exper)

print(len(vex_file.sources))
for source in vex_file.sources:
    print("   ", source)

print(len(vex_file.scans))
for scan in vex_file.scans:
    print("    ", scan)

print(len(vex_file.modes))
for mode in vex_file.modes:
    print("    ", mode)

print(len(vex_file.antennas))
for ant in vex_file.antennas:
    print("    ", ant)

print(len(vex_file.eops))
for eop in vex_file.eops:
    print("    ", eop)
```
