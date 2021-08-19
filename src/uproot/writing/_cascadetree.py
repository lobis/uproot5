# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
FIXME: docstring
"""

from __future__ import absolute_import

import datetime
import math
import struct

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy

import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization

_dtype_to_char = {
    numpy.dtype("bool"): "O",
    numpy.dtype(">i1"): "B",
    numpy.dtype(">u1"): "b",
    numpy.dtype(">i2"): "S",
    numpy.dtype(">u2"): "s",
    numpy.dtype(">i4"): "I",
    numpy.dtype(">u4"): "i",
    numpy.dtype(">i8"): "L",
    numpy.dtype(">u8"): "l",
    numpy.dtype(">f4"): "F",
    numpy.dtype(">f8"): "D",
}


class Tree(object):
    """
    FIXME: docstring
    """

    def __init__(
        self,
        directory,
        name,
        title,
        branch_types,
        freesegments,
        counter_name,
        field_name,
        initial_basket_capacity,
        resize_factor,
    ):
        self._directory = directory
        self._name = name
        self._title = title
        self._freesegments = freesegments
        self._counter_name = counter_name
        self._field_name = field_name
        self._basket_capacity = initial_basket_capacity
        self._resize_factor = resize_factor

        if isinstance(branch_types, dict):
            branch_types_items = branch_types.items()
        else:
            branch_types_items = branch_types

        if len(branch_types) == 0:
            raise ValueError("TTree must have at least one branch")

        self._branch_data = []
        self._branch_lookup = {}
        for branch_name, branch_type in branch_types_items:
            branch_dict = None
            branch_dtype = None
            branch_datashape = None

            if isinstance(branch_type, Mapping) and all(
                uproot._util.isstr(x) for x in branch_type
            ):
                branch_dict = branch_type

            else:
                try:
                    if type(branch_type).__module__.startswith("awkward."):
                        raise TypeError
                    if (
                        uproot._util.isstr(branch_type)
                        and branch_type.strip() == "bytes"
                    ):
                        raise TypeError
                    branch_dtype = numpy.dtype(branch_type)

                except TypeError:
                    try:
                        import awkward
                    except ImportError:
                        raise TypeError(
                            "not a NumPy dtype and 'awkward' cannot be imported: {0}".format(
                                repr(branch_type)
                            )
                        )
                    if isinstance(branch_type, awkward.types.Type):
                        branch_datashape = branch_type
                    else:
                        try:
                            branch_datashape = awkward.types.from_datashape(branch_type)
                        except Exception:
                            raise TypeError(
                                "not a NumPy dtype or an Awkward datashape: {0}".format(
                                    repr(branch_type)
                                )
                            )
                    # checking by class name to be Awkward v1/v2 insensitive
                    if type(branch_datashape).__name__ == "ArrayType":
                        if hasattr(branch_datashape, "content"):
                            branch_datashape = branch_datashape.content
                        else:
                            branch_datashape = branch_datashape.type
                    branch_dtype = self._branch_ak_to_np(branch_datashape)

            if branch_dict is not None:
                self._branch_lookup[branch_name] = len(self._branch_data)
                self._branch_data.append(
                    {"kind": "record", "name": branch_name, "keys": list(branch_dict)}
                )

                for key, content in branch_dict.items():
                    subname = self._field_name(branch_name, key)
                    try:
                        dtype = numpy.dtype(content)
                    except Exception:
                        raise TypeError(
                            "values of a dict must be NumPy types\n\n    key {0} has type {1}".format(
                                repr(key), repr(content)
                            )
                        )
                    self._branch_lookup[subname] = len(self._branch_data)
                    self._branch_data.append(self._branch_np(subname, content, dtype))

            elif branch_dtype is not None:
                self._branch_lookup[branch_name] = len(self._branch_data)
                self._branch_data.append(
                    self._branch_np(branch_name, branch_type, branch_dtype)
                )

            else:
                parameters = branch_datashape.parameters
                if parameters is None:
                    parameters = {}

                if parameters.get("__array__") == "string":
                    raise NotImplementedError("array of strings")

                elif parameters.get("__array__") == "bytes":
                    raise NotImplementedError("array of bytes")

                # checking by class name to be Awkward v1/v2 insensitive
                elif type(branch_datashape).__name__ == "ListType":
                    if hasattr(branch_datashape, "content"):
                        content = branch_datashape.content
                    else:
                        content = branch_datashape.type

                    counter_name = self._counter_name(branch_name)
                    counter_dtype = numpy.dtype(numpy.int32)
                    counter = self._branch_np(
                        counter_name, counter_dtype, counter_dtype, kind="counter"
                    )
                    self._branch_lookup[counter_name] = len(self._branch_data)
                    self._branch_data.append(counter)

                    if type(content).__name__ == "RecordType":
                        if hasattr(content, "contents"):
                            contents = content.contents
                        else:
                            contents = content.fields()
                        keys = content.keys
                        if callable(keys):
                            keys = keys()
                        if keys is None:
                            keys = [str(x) for x in range(len(contents))]

                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            {"kind": "record", "name": branch_name, "keys": keys}
                        )

                        for key, cont in zip(keys, contents):
                            subname = self._field_name(branch_name, key)
                            dtype = self._branch_ak_to_np(cont)
                            if dtype is None:
                                raise TypeError(
                                    "fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {0} has type {1}".format(
                                        repr(key), str(cont)
                                    )
                                )
                            self._branch_lookup[subname] = len(self._branch_data)
                            self._branch_data.append(
                                self._branch_np(subname, cont, dtype, counter=counter)
                            )

                    else:
                        dt = self._branch_ak_to_np(content)
                        if dt is None:
                            raise TypeError(
                                "cannot write Awkward Array type to ROOT file:\n\n    {0}".format(
                                    str(branch_datashape)
                                )
                            )
                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(branch_name, dt, dt, counter=counter)
                        )

                elif type(branch_datashape).__name__ == "RecordType":
                    if hasattr(branch_datashape, "contents"):
                        contents = branch_datashape.contents
                    else:
                        contents = branch_datashape.fields()
                    keys = branch_datashape.keys
                    if callable(keys):
                        keys = keys()
                    if keys is None:
                        keys = [str(x) for x in range(len(contents))]

                    self._branch_lookup[branch_name] = len(self._branch_data)
                    self._branch_data.append(
                        {"kind": "record", "name": branch_name, "keys": keys}
                    )

                    for key, content in zip(keys, contents):
                        subname = self._field_name(branch_name, key)
                        dtype = self._branch_ak_to_np(content)
                        if dtype is None:
                            raise TypeError(
                                "fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {0} has type {1}".format(
                                    repr(key), str(content)
                                )
                            )
                        self._branch_lookup[subname] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(subname, content, dtype)
                        )

                else:
                    raise TypeError(
                        "cannot write Awkward Array type to ROOT file:\n\n    {0}".format(
                            str(branch_datashape)
                        )
                    )

        self._num_entries = 0
        self._num_baskets = 0

        self._metadata_start = None
        self._metadata = {
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fSavedBytes": 0,
            "fFlushedBytes": 0,
            "fWeight": 1.0,
            "fTimerInterval": 0,
            "fScanField": 25,
            "fUpdate": 0,
            "fDefaultEntryOffsetLen": 1000,
            "fNClusterRange": 0,
            "fMaxEntries": 1000000000000,
            "fMaxEntryLoop": 1000000000000,
            "fMaxVirtualSize": 0,
            "fAutoSave": -300000000,
            "fAutoFlush": -30000000,
            "fEstimate": 1000000,
        }
        self._key = None

    def _branch_ak_to_np(self, branch_datashape):
        # checking by class name to be Awkward v1/v2 insensitive
        if type(branch_datashape).__name__ == "NumpyType":
            return numpy.dtype(branch_datashape.primitive)
        elif type(branch_datashape).__name__ == "PrimitiveType":
            return numpy.dtype(branch_datashape.dtype)
        elif type(branch_datashape).__name__ == "RegularType":
            if hasattr(branch_datashape, "content"):
                content = self._branch_ak_to_np(branch_datashape.content)
            else:
                content = self._branch_ak_to_np(branch_datashape.type)
            if content is None:
                return None
            elif content.subdtype is None:
                dtype, shape = content, ()
            else:
                dtype, shape = content.subdtype
            return numpy.dtype((dtype, (branch_datashape.size,) + shape))
        else:
            return None

    def _branch_np(
        self, branch_name, branch_type, branch_dtype, counter=None, kind="normal"
    ):
        branch_dtype = branch_dtype.newbyteorder(">")

        if branch_dtype.subdtype is None:
            branch_shape = ()
        else:
            branch_dtype, branch_shape = branch_dtype.subdtype

        letter = _dtype_to_char.get(branch_dtype)
        if letter is None:
            raise TypeError(
                "cannot write NumPy dtype {0} in TTree".format(branch_dtype)
            )

        if branch_shape == ():
            dims = ""
        else:
            dims = "".join("[" + str(x) + "]" for x in branch_shape)

        title = "{0}{1}/{2}".format(branch_name, dims, letter)

        return {
            "fName": branch_name,
            "branch_type": branch_type,
            "kind": kind,
            "counter": counter,
            "dtype": branch_dtype,
            "shape": branch_shape,
            "fTitle": title,
            "compression": self._directory.freesegments.fileheader.compression,
            "fBasketSize": 32000,
            "fEntryOffsetLen": 0 if counter is None else 1000,
            "fOffset": 0,
            "fSplitLevel": 0,
            "fFirstEntry": 0,
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fBasketBytes": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
            ),
            "fBasketEntry": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
            ),
            "fBasketSeek": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
            ),
            "metadata_start": None,
            "basket_metadata_start": None,
            "tleaf_reference_number": None,
            "tleaf_maximum_value": 0,
            "tleaf_special_struct": None,
        }

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6}, {7})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            [(datum["fName"], datum["branch_type"]) for datum in self._branch_data],
            self._freesegments,
            self._basket_capacity,
            self._resize_factor,
        )

    @property
    def directory(self):
        return self._directory

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._key.name

    @property
    def title(self):
        return self._key.title

    @property
    def branch_types(self):
        return self._branch_types

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def counter_name(self):
        return self._counter_name

    @property
    def field_name(self):
        return self._field_name

    @property
    def basket_capacity(self):
        return self._basket_capacity

    @property
    def resize_factor(self):
        return self._resize_factor

    @property
    def location(self):
        return self._key.location

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def num_baskets(self):
        return self._num_baskets

    def extend(self, file, sink, data):
        # expand capacity if this would REACH (not EXCEED) the existing capacity
        # that's because completely a full fBasketEntry has nowhere to put the
        # number of entries in the last basket (it's a fencepost principle thing),
        # forcing ROOT and Uproot to look it up from the basket header.
        if self._num_baskets >= self._basket_capacity - 1:
            self._basket_capacity = max(
                self._basket_capacity + 1,
                int(math.ceil(self._basket_capacity * self._resize_factor)),
            )

            for datum in self._branch_data:
                fBasketBytes = datum["fBasketBytes"]
                fBasketEntry = datum["fBasketEntry"]
                fBasketSeek = datum["fBasketSeek"]
                datum["fBasketBytes"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
                )
                datum["fBasketEntry"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
                )
                datum["fBasketSeek"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
                )
                datum["fBasketBytes"][: len(fBasketBytes)] = fBasketBytes
                datum["fBasketEntry"][: len(fBasketEntry)] = fBasketEntry
                datum["fBasketSeek"][: len(fBasketSeek)] = fBasketSeek
                datum["fBasketEntry"][len(fBasketEntry)] = self._num_entries

            oldloc = start = self._key.location
            stop = start + self._key.num_bytes + self._key.compressed_bytes

            self.write_anew(sink)

            newloc = self._key.seek_location
            file._move_tree(oldloc, newloc)

            self._freesegments.release(start, stop)
            sink.set_file_length(self._freesegments.fileheader.end)
            sink.flush()

        provided = None
        module_name = type(data).__module__

        if module_name == "pandas" or module_name.startswith("pandas."):
            import pandas

            if isinstance(data, pandas.DataFrame) and data.index.is_numeric():
                provided = dataframe_to_dict(data)

        if module_name == "awkward" or module_name.startswith("awkward."):
            import awkward

            if isinstance(data, awkward.Array):
                if data.ndim > 1 and not data.layout.purelist_isregular:
                    provided = {self._counter_name(""): awkward.num(data, axis=1)}
                else:
                    provided = {}
                for k, v in zip(awkward.fields(data), awkward.unzip(data)):
                    provided[k] = v

        if isinstance(data, numpy.ndarray) and data.dtype.fields is not None:
            provided = recarray_to_dict(data)

        if provided is None:
            if not isinstance(data, Mapping) or not all(
                uproot._util.isstr(x) for x in data
            ):
                raise TypeError(
                    "'extend' requires a mapping from branch name (str) to arrays"
                )

            provided = {}
            for k, v in data.items():
                module_name = type(v).__module__
                if module_name == "awkward" or module_name.startswith("awkward."):
                    import awkward

                    if (
                        isinstance(v, awkward.Array)
                        and v.ndim > 1
                        and not v.layout.purelist_isregular
                    ):
                        provided[self._counter_name(k)] = awkward.num(v, axis=1)

                provided[k] = v

        actual_branches = {}
        for datum in self._branch_data:
            if datum["kind"] == "record":
                if datum["name"] in provided:
                    recordarray = provided.pop(datum["name"])

                    module_name = type(recordarray).__module__
                    if module_name == "pandas" or module_name.startswith("pandas."):
                        import pandas

                        if isinstance(recordarray, pandas.DataFrame):
                            tmp = {"index": recordarray.index.values}
                            for column in recordarray.columns:
                                tmp[column] = recordarray[column]
                            recordarray = tmp

                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = recordarray[
                            key
                        ]

                elif datum["name"] == "":
                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = provided.pop(
                            key
                        )

                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {0}".format(
                            repr(datum["name"])
                        )
                    )

            else:
                if datum["fName"] in provided:
                    actual_branches[datum["fName"]] = provided.pop(datum["fName"])
                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {0}".format(
                            repr(datum["fName"])
                        )
                    )

        if len(provided) != 0:
            raise ValueError(
                "'extend' was given data that do not correspond to any branch: {0}".format(
                    ", ".join(repr(x) for x in provided)
                )
            )

        tofill = []
        num_entries = None
        for branch_name, branch_array in actual_branches.items():
            if num_entries is None:
                num_entries = len(branch_array)
            elif num_entries != len(branch_array):
                raise ValueError(
                    "'extend' must fill every branch with the same number of entries; {0} has {1} entries".format(
                        repr(branch_name),
                        len(branch_array),
                    )
                )

            datum = self._branch_data[self._branch_lookup[branch_name]]
            if datum["kind"] == "record":
                continue

            if datum["counter"] is None:
                big_endian = numpy.asarray(branch_array, dtype=datum["dtype"])
                if big_endian.shape != (len(branch_array),) + datum["shape"]:
                    raise ValueError(
                        "'extend' must fill branches with a consistent shape: has {0}, trying to fill with {1}".format(
                            datum["shape"],
                            big_endian.shape[1:],
                        )
                    )
                tofill.append((branch_name, big_endian, None))

                if datum["kind"] == "counter":
                    datum["tleaf_maximum_value"] = max(
                        big_endian.max(), datum["tleaf_maximum_value"]
                    )

            else:
                import awkward

                layout = branch_array.layout
                while not isinstance(
                    layout,
                    (
                        awkward.layout.ListOffsetArray32,
                        awkward.layout.ListOffsetArrayU32,
                        awkward.layout.ListOffsetArray64,
                    ),
                ):
                    if isinstance(layout, awkward.partition.PartitionedArray):
                        layout = awkward.concatenate(layout.partitions, highlevel=False)

                    elif isinstance(
                        layout,
                        (
                            awkward.layout.IndexedArray32,
                            awkward.layout.IndexedArrayU32,
                            awkward.layout.IndexedArray64,
                        ),
                    ):
                        layout = layout.project()

                    elif isinstance(
                        layout,
                        (
                            awkward.layout.ListArray32,
                            awkward.layout.ListArrayU32,
                            awkward.layout.ListArray64,
                        ),
                    ):
                        layout = layout.toListOffsetArray64(False)

                    else:
                        raise AssertionError(
                            "how did this pass the type check?\n\n" + repr(layout)
                        )

                content = layout.content
                offsets = numpy.asarray(layout.offsets)
                if offsets[0] != 0:
                    content = content[offsets[0] :]
                    offsets -= offsets[0]
                if len(content) > offsets[-1]:
                    content = content[: offsets[-1]]

                shape = [len(content)]
                while not isinstance(content, awkward.layout.NumpyArray):
                    if isinstance(
                        content,
                        (
                            awkward.layout.IndexedArray32,
                            awkward.layout.IndexedArrayU32,
                            awkward.layout.IndexedArray64,
                        ),
                    ):
                        content = content.project()

                    elif isinstance(content, awkward.layout.EmptyArray):
                        content = content.toNumpyArray()

                    elif isinstance(content, awkward.layout.RegularArray):
                        shape.append(content.size)
                        content = content.content

                    else:
                        raise AssertionError(
                            "how did this pass the type check?\n\n" + repr(content)
                        )

                big_endian = numpy.asarray(content, dtype=datum["dtype"])
                shape = tuple(shape) + big_endian.shape[1:]

                if shape[1:] != datum["shape"]:
                    raise ValueError(
                        "'extend' must fill branches with a consistent shape: has {0}, trying to fill with {1}".format(
                            datum["shape"],
                            shape[1:],
                        )
                    )
                big_endian_offsets = offsets.astype(">i4", copy=True)

                tofill.append((branch_name, big_endian.reshape(-1), big_endian_offsets))

        # actually write baskets into the file
        uncompressed_bytes = 0
        compressed_bytes = 0
        for branch_name, big_endian, big_endian_offsets in tofill:
            datum = self._branch_data[self._branch_lookup[branch_name]]

            if big_endian_offsets is None:
                totbytes, zipbytes, location = self.write_np_basket(
                    sink, branch_name, big_endian
                )
            else:
                totbytes, zipbytes, location = self.write_jagged_basket(
                    sink, branch_name, big_endian, big_endian_offsets
                )
                datum["fEntryOffsetLen"] = 4 * (len(big_endian_offsets) - 1)
            uncompressed_bytes += totbytes
            compressed_bytes += zipbytes

            datum["fTotBytes"] += totbytes
            datum["fZipBytes"] += zipbytes

            datum["fBasketBytes"][self._num_baskets] = zipbytes

            if self._num_baskets + 1 < self._basket_capacity:
                fBasketEntry = datum["fBasketEntry"]
                i = self._num_baskets
                fBasketEntry[i + 1] = num_entries + fBasketEntry[i]

            datum["fBasketSeek"][self._num_baskets] = location

        # update TTree metadata in file
        self._num_entries += num_entries
        self._num_baskets += 1
        self._metadata["fTotBytes"] += uncompressed_bytes
        self._metadata["fZipBytes"] += compressed_bytes

        self.write_updates(sink)

    def write_anew(self, sink):
        key_num_bytes = uproot.reading._key_format_big.size + 6
        name_asbytes = self._name.encode(errors="surrogateescape")
        title_asbytes = self._title.encode(errors="surrogateescape")
        key_num_bytes += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        key_num_bytes += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        out = [None]
        ttree_header_index = 0

        tobject = uproot.models.TObject.Model_TObject.empty()
        tnamed = uproot.models.TNamed.Model_TNamed.empty()
        tnamed._bases.append(tobject)
        tnamed._members["fTitle"] = self._title
        tnamed._serialize(out, True, self._name, uproot.const.kMustCleanup)

        # TAttLine v2, fLineColor: 602 fLineStyle: 1 fLineWidth: 1
        # TAttFill v2, fFillColor: 0, fFillStyle: 1001
        # TAttMarker v2, fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1.0
        out.append(
            b"@\x00\x00\x08\x00\x02\x02Z\x00\x01\x00\x01"
            + b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9"
            + b"@\x00\x00\n\x00\x02\x00\x01\x00\x01?\x80\x00\x00"
        )

        metadata_out_index = len(out)
        out.append(
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            )
        )

        # speedbump (0), fClusterRangeEnd (empty array),
        # speedbump (0), fClusterSize (empty array)
        # fIOFeatures (TIOFeatures)
        out.append(b"\x00\x00@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

        tleaf_reference_numbers = []

        tobjarray_of_branches_index = len(out)
        out.append(None)

        num_branches = sum(
            0 if datum["kind"] == "record" else 1 for datum in self._branch_data
        )

        # TObjArray header with fName: ""
        out.append(b"\x00\x01\x00\x00\x00\x00\x03\x00@\x00\x00")
        out.append(
            uproot.models.TObjArray._tobjarray_format1.pack(
                num_branches,  # TObjArray fSize
                0,  # TObjArray fLowerBound
            )
        )

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            any_tbranch_index = len(out)
            out.append(None)
            out.append(b"TBranch\x00")

            tbranch_index = len(out)
            out.append(None)

            tbranch_tobject = uproot.models.TObject.Model_TObject.empty()
            tbranch_tnamed = uproot.models.TNamed.Model_TNamed.empty()
            tbranch_tnamed._bases.append(tbranch_tobject)
            tbranch_tnamed._members["fTitle"] = datum["fTitle"]
            tbranch_tnamed._serialize(
                out, True, datum["fName"], numpy.uint32(0x00400000)
            )

            # TAttFill v2, fFillColor: 0, fFillStyle: 1001
            out.append(b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9")

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            if datum["compression"] is None:
                fCompress = uproot.compression.ZLIB(0).code
            else:
                fCompress = datum["compression"].code

            out.append(
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self._num_baskets,  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                )
            )

            # fIOFeatures (TIOFeatures)
            out.append(b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

            out.append(
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    self._basket_capacity,  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                )
            )

            # empty TObjArray of TBranches
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            subtobjarray_of_leaves_index = len(out)
            out.append(None)

            # TObjArray header with fName: "", fSize: 1, fLowerBound: 0
            out.append(
                b"\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00"
            )

            absolute_location = key_num_bytes + sum(
                len(x) for x in out if x is not None
            )
            absolute_location += 8 + 6 * (sum(1 if x is None else 0 for x in out) - 1)
            datum["tleaf_reference_number"] = absolute_location + 2
            tleaf_reference_numbers.append(datum["tleaf_reference_number"])

            subany_tleaf_index = len(out)
            out.append(None)

            letter = _dtype_to_char[datum["dtype"]]
            letter_upper = letter.upper()
            out.append(("TLeaf" + letter_upper).encode() + b"\x00")
            if letter_upper == "O":
                special_struct = uproot.models.TLeaf._tleafO1_format1
            elif letter_upper == "B":
                special_struct = uproot.models.TLeaf._tleafb1_format1
            elif letter_upper == "S":
                special_struct = uproot.models.TLeaf._tleafs1_format1
            elif letter_upper == "I":
                special_struct = uproot.models.TLeaf._tleafi1_format1
            elif letter_upper == "L":
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif letter_upper == "F":
                special_struct = uproot.models.TLeaf._tleaff1_format1
            elif letter_upper == "D":
                special_struct = uproot.models.TLeaf._tleafd1_format1
            fLenType = datum["dtype"].itemsize
            fIsUnsigned = letter != letter_upper

            if datum["shape"] == ():
                dims = ""
            else:
                dims = "".join("[" + str(x) + "]" for x in datum["shape"])

            # single TLeaf
            leaf_name = datum["fName"].encode(errors="surrogateescape")
            leaf_title = (datum["fName"] + dims).encode(errors="surrogateescape")
            leaf_name_length = (1 if len(leaf_name) < 255 else 5) + len(leaf_name)
            leaf_title_length = (1 if len(leaf_title) < 255 else 5) + len(leaf_title)

            leaf_header = numpy.array(
                [64, 0, 0, 76, 0, 1, 64, 0, 0, 54, 0, 2, 64, 0]
                + [0, 30, 0, 1, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0],
                numpy.uint8,
            )
            tmp = leaf_header[0:4].view(">u4")
            tmp[:] = (
                numpy.uint32(
                    42 + leaf_name_length + leaf_title_length + special_struct.size
                )
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[6:10].view(">u4")
            tmp[:] = (
                numpy.uint32(36 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[12:16].view(">u4")
            tmp[:] = (
                numpy.uint32(12 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )

            out.append(uproot._util.tobytes(leaf_header))
            if len(leaf_name) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_name), len(leaf_name), leaf_name)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_name), 255, len(leaf_name), leaf_name
                    )
                )
            if len(leaf_title) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_title), len(leaf_title), leaf_title)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_title), 255, len(leaf_title), leaf_title
                    )
                )

            fLen = 1
            for item in datum["shape"]:
                fLen *= item

            # generic TLeaf members
            out.append(
                uproot.models.TLeaf._tleaf2_format0.pack(
                    fLen,
                    fLenType,
                    0,  # fOffset
                    datum["kind"] == "counter",  # fIsRange
                    fIsUnsigned,
                )
            )

            if datum["counter"] is None:
                # null fLeafCount
                out.append(b"\x00\x00\x00\x00")
            else:
                # reference to fLeafCount
                out.append(
                    uproot.deserialization._read_object_any_format1.pack(
                        datum["counter"]["tleaf_reference_number"]
                    )
                )

            # specialized TLeaf* members (fMinimum, fMaximum)
            out.append(special_struct.pack(0, 0))
            datum["tleaf_special_struct"] = special_struct

            out[
                subany_tleaf_index
            ] = uproot.serialization._serialize_object_any_format1.pack(
                numpy.uint32(sum(len(x) for x in out[subany_tleaf_index + 1 :]) + 4)
                | uproot.const.kByteCountMask,
                uproot.const.kNewClassTag,
            )

            out[subtobjarray_of_leaves_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[subtobjarray_of_leaves_index + 1 :]),
                3,  # TObjArray
            )

            # empty TObjArray of fBaskets (embedded)
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["basket_metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # speedbump and fBasketBytes
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketBytes"]))

            # speedbump and fBasketEntry
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketEntry"]))

            # speedbump and fBasketSeek
            out.append(b"\x01")
            out.append(uproot._util.tobytes(datum["fBasketSeek"]))

            # empty fFileName
            out.append(b"\x00")

            out[tbranch_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[tbranch_index + 1 :]), 13  # TBranch
            )

            out[
                any_tbranch_index
            ] = uproot.serialization._serialize_object_any_format1.pack(
                numpy.uint32(sum(len(x) for x in out[any_tbranch_index + 1 :]) + 4)
                | uproot.const.kByteCountMask,
                uproot.const.kNewClassTag,
            )

        out[tobjarray_of_branches_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[tobjarray_of_branches_index + 1 :]), 3  # TObjArray
        )

        # TObjArray of TLeaf references
        tleaf_reference_bytes = uproot._util.tobytes(
            numpy.array(tleaf_reference_numbers, ">u4")
        )
        out.append(
            struct.pack(
                ">I13sI4s",
                (21 + len(tleaf_reference_bytes)) | uproot.const.kByteCountMask,
                b"\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00",
                len(tleaf_reference_numbers),
                b"\x00\x00\x00\x00",
            )
        )

        out.append(tleaf_reference_bytes)

        # null fAliases (b"\x00\x00\x00\x00")
        # empty fIndexValues array (4-byte length is zero)
        # empty fIndex array (4-byte length is zero)
        # null fTreeIndex (b"\x00\x00\x00\x00")
        # null fFriends (b"\x00\x00\x00\x00")
        # null fUserInfo (b"\x00\x00\x00\x00")
        # null fBranchRef (b"\x00\x00\x00\x00")
        out.append(b"\x00" * 28)

        out[ttree_header_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[ttree_header_index + 1 :]), 20  # TTree
        )

        self._metadata_start = sum(len(x) for x in out[:metadata_out_index])

        raw_data = b"".join(out)
        self._key = self._directory.add_object(
            sink,
            "TTree",
            self._name,
            self._title,
            raw_data,
            len(raw_data),
            replaces=self._key,
            big=True,
        )

    def write_updates(self, sink):
        base = self._key.seek_location + self._key.num_bytes

        sink.write(
            base + self._metadata_start,
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            ),
        )
        sink.flush()

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            position = base + datum["metadata_start"]

            if datum["compression"] is None:
                fCompress = uproot.compression.ZLIB(0).code
            else:
                fCompress = datum["compression"].code

            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self._num_baskets,  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                ),
            )

            position += uproot.models.TBranch._tbranch13_format1.size + 11
            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    self._basket_capacity,  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                ),
            )

            fBasketBytes = uproot._util.tobytes(datum["fBasketBytes"])
            fBasketEntry = uproot._util.tobytes(datum["fBasketEntry"])
            fBasketSeek = uproot._util.tobytes(datum["fBasketSeek"])

            position = base + datum["basket_metadata_start"] + 1
            sink.write(position, fBasketBytes)
            position += len(fBasketBytes) + 1
            sink.write(position, fBasketEntry)
            position += len(fBasketEntry) + 1
            sink.write(position, fBasketSeek)

            if datum["kind"] == "counter":
                position = (
                    base
                    + datum["basket_metadata_start"]
                    - 25  # empty TObjArray of fBaskets (embedded)
                    - datum["tleaf_special_struct"].size
                )
                sink.write(
                    position,
                    datum["tleaf_special_struct"].pack(
                        0,
                        datum["tleaf_maximum_value"],
                    ),
                )

    def write_np_basket(self, sink, branch_name, array):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        raw_array = uproot._util.tobytes(array)
        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item

        fObjlen = len(raw_array)

        fNbytes = fKeylen + fObjlen  # FIXME: no compression yet

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                itemsize,  # fNevBufSize
                len(array),  # fNevBuf
                fKeylen + len(raw_array),  # fLast
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(raw_array)

        sink.write(location, b"".join(out))
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fNbytes, fNbytes, location

    def write_jagged_basket(self, sink, branch_name, array, offsets):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        raw_array = uproot._util.tobytes(array)
        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item

        # offsets became a *copy* of the Awkward Array's offsets
        # when it was converted to big-endian (astype with copy=True)
        offsets *= itemsize
        offsets += fKeylen
        fLast = offsets[-1]
        offsets[-1] = 0
        raw_offsets = uproot._util.tobytes(offsets)

        fObjlen = len(raw_array) + 4 + len(raw_offsets)

        fNbytes = fKeylen + fObjlen  # FIXME: no compression yet

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                len(offsets) + 1,  # fNevBufSize
                len(offsets) - 1,  # fNevBuf
                fLast,
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(raw_array)
        out.append(_tbasket_offsets_length.pack(len(offsets)))
        out.append(raw_offsets)

        sink.write(location, b"".join(out))
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fNbytes, fNbytes, location


_tbasket_offsets_length = struct.Struct(">I")


def dataframe_to_dict(df):
    """
    FIXME: docstring
    """
    out = {"index": df.index.values}
    for column_name in df.columns:
        out[str(column_name)] = df[column_name].values
    return out


def recarray_to_dict(array):
    """
    FIXME: docstring
    """
    out = {}
    for field_name in array.dtype.fields:
        field = array[field_name]
        if field.dtype.fields is not None:
            for subfield_name, subfield in recarray_to_dict(field):
                out[field_name + "." + subfield_name] = subfield
        else:
            out[field_name] = field
    return out