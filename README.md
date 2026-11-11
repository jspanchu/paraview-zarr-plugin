# paraview-zarr-plugin

A ParaView Python plugin that reads [Zarr](https://zarr.dev) stores (v2 and v3,
including OME-Zarr) as image data.

- All numeric arrays on the same grid -> point-data arrays on vtkImageData.
- Honors OME-Zarr axes, scale, translation, channel labels and pyramid levels.
- time axis becomes ParaView time steps.
- reads only requested sub-extent and time step.

## Setup

Requires ParaView 6.2 or later, which can manage `uv`-backed virtual
environments for its bundled Python.

Create an environment named `zarr` with the reader's dependencies:

```sh
pvpython -m paraview.envs create zarr ./requirements.txt
```

## Usage

Start ParaView with that environment and this directory on the plugin path.
Plugins found in `PV_PLUGIN_PATH` are loaded automatically.

```sh
export PV_PLUGIN_PATH=/path/to/zarr-pv-plugin
paraview --venv=$HOME/.config/ParaView/uv-venvs/3.12/zarr/venv
```

Then open a `*.zarr` directory with File > Open. The same options work for
`pvpython`, `pvbatch` and `pvserver`:

```python
from paraview.simple import *

reader = ZarrImageReader(FileName="image.zarr")
reader.ResolutionLevel = 0
Show(reader)
```
