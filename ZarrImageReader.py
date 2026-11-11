"""ParaView reader plugin for image data stored in Zarr (v2 or v3) stores.

The reader opens a ``*.zarr`` directory store (or, from Python, a zipped
store) and produces one vtkImageData. Every numeric array in the store that
lives on the same grid becomes a point-data array.

Axis meaning is taken in this order of preference:
  1. OME-Zarr (NGFF) ``multiscales`` metadata: axes, scale and translation.
     The "Resolution Level" property picks the pyramid level.
  2. Zarr v3 ``dimension_names`` / xarray ``_ARRAY_DIMENSIONS``, when the
     names are x, y, z, t/time, c/channel.
  3. Position: (y, x), (z, y, x), (c, z, y, x) or (t, c, z, y, x).

Without OME metadata, origin and spacing come from ``origin`` / ``spacing``
attributes (a scalar or three values in x, y, z order) on the array or on the
store root, defaulting to 0 and 1.

A time axis becomes ParaView time steps. A channel axis is split into one
point-data array per channel, named from the OME ``omero`` channel labels if
present, else "<name>_0", "<name>_1", ...

Only the requested sub-extent and time step are read from the store, so
chunked stores larger than memory can be sliced or served in parallel.

Requires the ``zarr`` Python package (>= 3) from a virtual environment:
    paraview --venv=/path/to/venv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smdomain,
    smhint,
    smproperty,
    smproxy,
)
from vtkmodules.vtkCommonCore import vtkDataArraySelection
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonExecutionModel import (
    vtkAlgorithm,
    vtkStreamingDemandDrivenPipeline,
)
from vtkmodules.util import numpy_support

paraview_plugin_name = "ZarrImageReader"
paraview_plugin_version = "1.0"

_AXIS_ALIASES = {
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "time": "t",
    "c": "c",
    "channel": "c",
    "channels": "c",
}
_POSITIONAL_AXES = {2: "yx", 3: "zyx", 4: "czyx", 5: "tczyx"}
_METADATA_FILES = ("zarr.json", ".zgroup", ".zarray", ".zattrs", ".zmetadata")


def _import_zarr():
    """Import and return the ``zarr`` module.

    Returns
    -------
    module
        The imported ``zarr`` package.

Raises
    ------
    RuntimeError
        If ``zarr`` cannot be imported. ParaView does not bundle it, so the
        message tells the user how to provide it.
    """
    try:
        import zarr
    except ImportError as err:
        raise RuntimeError(
            "The Zarr Image Reader needs the 'zarr' Python package, which could "
            f"not be imported ({err}). Start ParaView with a virtual environment "
            "that provides it: `paraview --venv /path/to/venv`."
        ) from err
    return zarr


def _open_store(path: str):
    """Open a directory or zip store read-only.

    Parameters
    ----------
    path : str
        A store directory, a zipped store, or one of the store's metadata
        files (which selects the directory that contains it).

    Returns
    -------
    zarr.Group or zarr.Array
        The root node of the store.
    """
    zarr = _import_zarr()
    store = Path(path)
    if store.is_file():
        if store.name in _METADATA_FILES:
            # Picking the metadata file in a file dialog means its directory.
            store = store.parent
        else:
            return zarr.open(zarr.storage.ZipStore(store, mode="r"), mode="r")
    return zarr.open(store, mode="r")


def _axes(names: list | None, ndim: int) -> list[str] | None:
    """Work out the axis roles of an array.

    Parameters
    ----------
    names : list of str or None
        Axis names from the store metadata, if any.
    ndim : int
        Number of dimensions of the array.

    Returns
    -------
    list of str or None
        One of 'x', 'y', 'z', 't', 'c' per dimension. Falls back to the
        positional convention when ``names`` is missing or unusable. None when
        neither applies.
    """
    if names and len(names) == ndim:
        roles = [_AXIS_ALIASES.get(str(n).lower()) for n in names]
        if None not in roles and len(set(roles)) == ndim and {"x", "y"} <= set(roles):
            return roles
    positional = _POSITIONAL_AXES.get(ndim)
    return list(positional) if positional else None


def _ome_datasets(groups: dict) -> dict[str, dict]:
    """Collect per-array OME-Zarr metadata.

    Parameters
    ----------
    groups : dict
        Maps a group path to its ``zarr.Group``.

    Returns
    -------
    dict
        Maps an array path to a dict with the keys ``names``, ``scale``,
        ``translation``, ``level``, ``nlevels``, ``name`` and ``labels``.
    """
    datasets = {}
    for gpath, group in groups.items():
        attrs = group.attrs.asdict()
        attrs = attrs.get("ome", attrs)  # NGFF 0.5 nests under "ome"
        multiscales = attrs.get("multiscales")
        if not isinstance(multiscales, list):
            continue
        channels = (attrs.get("omero") or {}).get("channels") or []
        labels = [str(ch.get("label", i)) for i, ch in enumerate(channels)]
        for ms in multiscales:
            axes = ms.get("axes") or []
            names = [a["name"] if isinstance(a, dict) else a for a in axes]
            levels = ms.get("datasets") or []
            for level, ds in enumerate(levels):
                scale = translation = None
                for tf in ds.get("coordinateTransformations") or []:
                    if tf.get("type") == "scale":
                        scale = tf.get("scale")
                    elif tf.get("type") == "translation":
                        translation = tf.get("translation")
                path = "/".join(p for p in (gpath, ds.get("path", "")) if p)
                datasets[path] = dict(
                    names=names,
                    scale=scale,
                    translation=translation,
                    level=level,
                    nlevels=len(levels),
                    name=ms.get("name") or gpath,
                    labels=labels,
                )
    return datasets


def _xyz(value, default: float) -> list[float]:
    """Coerce a scalar or 3-sequence attribute to an [x, y, z] list.

    Parameters
    ----------
    value : float or sequence of float
        The attribute value. A scalar applies to all three axes.
    default : float
        Value used for all three axes when ``value`` is malformed.

    Returns
    -------
    list of float
        Three values in x, y, z order.
    """
    try:
        values = np.atleast_1d(np.asarray(value, dtype=float)).tolist()
    except (TypeError, ValueError):
        return [default] * 3
    if len(values) == 1:
        return values * 3
    return values if len(values) == 3 else [default] * 3


@smproxy.reader(
    name="ZarrImageReader",
    label="Zarr Image Reader",
    extensions="zarr",
    file_description="Zarr image stores",
    is_directory=True,
)
class ZarrImageReader(VTKPythonAlgorithmBase):
    """Reads image arrays from a Zarr store into a vtkImageData."""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkImageData"
        )
        self._filename = None
        self._level = 0
        self._meta = None
        self._arrayselection = vtkDataArraySelection()
        self._arrayselection.AddObserver("ModifiedEvent", lambda *_: self.Modified())

    def _read_metadata(self) -> dict:
        """Open the store and work out the grid, time steps and point arrays.

        The result is cached until the file name or resolution level changes.

        Returns
        -------
        dict
            The keys ``dims`` (nx, ny, nz), ``origin``, ``spacing``, ``times``
            (list of float or None) and ``fields``, a list of
            ``(vtk name, zarr array, axes, channel index or None)``.

        Raises
        ------
        RuntimeError
            If no file name is set or the store has no usable image array.
        """
        if self._meta is not None:
            return self._meta
        if not self._filename:
            raise RuntimeError("FileName has not been set.")

        root = _open_store(self._filename)
        stem = Path(self._filename).name.split(".")[0]
        if hasattr(root, "members"):
            groups, arrays = {"": root}, {}
            for path, node in root.members(max_depth=None):
                (groups if hasattr(node, "members") else arrays)[path] = node
        else:
            groups, arrays = {}, {"": root}
        ome = _ome_datasets(groups)

        # candidates: path -> dict(array, axes, shape (nx, ny, nz), nt, nc)
        candidates = {}
        for path in sorted(arrays):
            array = arrays[path]
            info = ome.get(path)
            if info and info["level"] != min(self._level, info["nlevels"] - 1):
                continue
            if array.dtype.kind not in "biuf":
                continue
            attrs = array.attrs.asdict()
            names = (
                (info and info["names"])
                or getattr(array.metadata, "dimension_names", None)
                or attrs.get("_ARRAY_DIMENSIONS")
            )
            axes = _axes(names, array.ndim)
            if axes is None:
                continue
            size = dict(zip(axes, array.shape))
            candidates[path] = dict(
                array=array,
                axes=axes,
                attrs=attrs,
                shape=(size["x"], size["y"], size.get("z", 1)),
                nt=size.get("t"),
                nc=size.get("c"),
            )
        if not candidates:
            raise RuntimeError(f"No image arrays found in {self._filename}")

        # The grid is defined by the first OME image, else by the largest array.
        ref_path = next(
            (p for p in candidates if p in ome),
            max(candidates, key=lambda p: candidates[p]["array"].size),
        )
        ref = candidates[ref_path]
        dims, nt = ref["shape"], ref["nt"]

        spacing, origin, times = [1.0] * 3, [0.0] * 3, None
        info = ome.get(ref_path)
        if info:
            scale = dict(zip(ref["axes"], info["scale"] or []))
            shift = dict(zip(ref["axes"], info["translation"] or []))
            spacing = [float(scale.get(a) or 1.0) for a in "xyz"]
            origin = [float(shift.get(a) or 0.0) for a in "xyz"]
            if nt:
                dt, t0 = float(scale.get("t") or 1.0), float(shift.get("t") or 0.0)
                times = [t0 + i * dt for i in range(nt)]
        else:
            # The array's own attributes override those of the store root.
            root_attrs = root.attrs.asdict()
            for attrs in (root_attrs, ref["attrs"]):
                if "spacing" in attrs:
                    spacing = _xyz(attrs["spacing"], 1.0)
                if "origin" in attrs:
                    origin = _xyz(attrs["origin"], 0.0)
        if nt and times is None:
            times = [float(i) for i in range(nt)]

        # fields: list of (vtk name, array, axes, channel index or None)
        fields, used = [], set()

        def add(name, cand, channel):
            # VTK point arrays are looked up by name, so names must be unique.
            base, n = name, 1
            while name in used:
                n += 1
                name = f"{base}_{n}"
            used.add(name)
            fields.append((name, cand["array"], cand["axes"], channel))

        for path, cand in candidates.items():
            if cand["shape"] != dims or (cand["nt"] and cand["nt"] != nt):
                continue
            info = ome.get(path)
            name = (info["name"] if info else path) or stem
            if cand["nc"] is None:
                add(name, cand, None)
            elif info and len(info["labels"]) == cand["nc"]:
                for c, label in enumerate(info["labels"]):
                    add(label, cand, c)
            elif cand["nc"] == 1:
                add(name, cand, 0)
            else:
                for c in range(cand["nc"]):
                    add(f"{name}_{c}", cand, c)

        for name, *_ in fields:
            if not self._arrayselection.ArrayExists(name):
                self._arrayselection.AddArray(name)

        self._meta = dict(
            dims=dims, origin=origin, spacing=spacing, times=times, fields=fields
        )
        return self._meta

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="zarr", file_description="Zarr image stores")
    def SetFileName(self, name: str):
        """Set the Zarr store to read.

        Parameters
        ----------
        name : str
            Path of the store directory (or zipped store).
        """
        if self._filename != name:
            self._filename = name
            self._meta = None
            self._arrayselection.RemoveAllArrays()
            self.Modified()

    @smproperty.intvector(
        name="ResolutionLevel", label="Resolution Level", default_values=0
    )
    @smdomain.intrange(min=0, max=16)
    def SetResolutionLevel(self, level: int):
        """OME-Zarr pyramid level to read; 0 is full resolution.

        Parameters
        ----------
        level : int
            Pyramid level. Negative values clamp to 0, and values past the
            coarsest level read the coarsest level.
        """
        level = max(0, int(level))
        if self._level != level:
            self._level = level
            self._meta = None
            self._arrayselection.RemoveAllArrays()
            self.Modified()

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self) -> list[float] | None:
        """Get the time values of the store.

        Returns
        -------
        list of float or None
            One value per time step. None when no file is set or the store has
            no time axis.
        """
        return self._read_metadata()["times"] if self._filename else None

    @smproperty.dataarrayselection(name="Point Arrays")
    def GetPointDataArraySelection(self) -> vtkDataArraySelection:
        """Get the selection that enables or disables each point array.

        Returns
        -------
        vtkDataArraySelection
            The selection, filled in once the store has been inspected.
        """
        return self._arrayselection

    def RequestInformation(self, request, inInfo, outInfo) -> int:
        """Publish the whole extent, geometry and time steps.

        Parameters
        ----------
        request : vtkInformation
            The pipeline request. Unused.
        inInfo : list of vtkInformationVector
            Input information. Unused, the reader has no input ports.
        outInfo : vtkInformationVector
            Output information to fill in.

        Returns
        -------
        int
            1 on success.
        """
        meta = self._read_metadata()
        nx, ny, nz = meta["dims"]
        sddp = vtkStreamingDemandDrivenPipeline
        info = outInfo.GetInformationObject(0)
        info.Set(sddp.WHOLE_EXTENT(), (0, nx - 1, 0, ny - 1, 0, nz - 1), 6)
        info.Set(vtkImageData.SPACING(), meta["spacing"], 3)
        info.Set(vtkImageData.ORIGIN(), meta["origin"], 3)
        info.Set(vtkAlgorithm.CAN_PRODUCE_SUB_EXTENT(), 1)
        # Clear stale values left over from a previous file or level.
        info.Remove(sddp.TIME_STEPS())
        info.Remove(sddp.TIME_RANGE())
        if meta["times"]:
            times = meta["times"]
            info.Set(sddp.TIME_STEPS(), times, len(times))
            info.Set(sddp.TIME_RANGE(), [times[0], times[-1]], 2)
        return 1

    def RequestData(self, request, inInfo, outInfo) -> int:
        """Read the requested sub-extent and time step into the output.

        Parameters
        ----------
        request : vtkInformation
            The pipeline request. Unused.
        inInfo : list of vtkInformationVector
            Input information. Unused, the reader has no input ports.
        outInfo : vtkInformationVector
            Output information holding the update extent and time step.

        Returns
        -------
        int
            1 on success.
        """
        meta = self._read_metadata()
        sddp = vtkStreamingDemandDrivenPipeline
        info = outInfo.GetInformationObject(0)
        extent = list(info.Get(sddp.UPDATE_EXTENT()))

        tindex = 0
        if meta["times"]:
            times = np.asarray(meta["times"])
            if info.Has(sddp.UPDATE_TIME_STEP()):
                tindex = int(np.abs(times - info.Get(sddp.UPDATE_TIME_STEP())).argmin())

        output = vtkImageData.GetData(outInfo, 0)
        output.SetExtent(extent)
        output.SetOrigin(meta["origin"])
        output.SetSpacing(meta["spacing"])
        if meta["times"]:
            output.GetInformation().Set(output.DATA_TIME_STEP(), meta["times"][tindex])
        if any(extent[2 * i + 1] < extent[2 * i] for i in range(3)):
            return 1

        index = {
            "x": slice(extent[0], extent[1] + 1),
            "y": slice(extent[2], extent[3] + 1),
            "z": slice(extent[4], extent[5] + 1),
            "t": tindex,
        }
        pd = output.GetPointData()
        for name, array, axes, channel in meta["fields"]:
            if not self._arrayselection.ArrayIsEnabled(name):
                continue
            index["c"] = channel
            block = array[tuple(index[a] for a in axes)]
            # Reorder the remaining spatial axes to (z, y, x): x varies fastest.
            spatial = [a for a in axes if a in "xyz"]
            order = sorted(range(len(spatial)), key=lambda i: "zyx".index(spatial[i]))
            block = np.transpose(block, order)
            # VTK has no bool or float16 arrays, and expects native byte order.
            if block.dtype == np.bool_:
                block = block.astype(np.uint8)
            elif block.dtype == np.float16:
                block = block.astype(np.float32)
            elif not block.dtype.isnative:
                block = block.astype(block.dtype.newbyteorder("="))
            flat = np.ascontiguousarray(block).ravel()
            vtkarr = numpy_support.numpy_to_vtk(flat, deep=1)
            vtkarr.SetName(name)
            pd.AddArray(vtkarr)

        if pd.GetNumberOfArrays() > 0:
            pd.SetActiveScalars(pd.GetArrayName(0))
        return 1
