# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::Experimental::RNTuple``.
"""
from __future__ import annotations

import struct
import zlib

import numpy

import uproot

# https://github.com/root-project/root/blob/e9fa243af91217e9b108d828009c81ccba7666b5/tree/ntuple/v7/inc/ROOT/RMiniFile.hxx#L65
_rntuple_format1 = struct.Struct(">IIIQIIQIIQ")

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#envelopes
_rntuple_feature_flag_format = struct.Struct("<Q")
_rntuple_num_bytes_fields = struct.Struct("<II")
_rntuple_field_description = struct.Struct("<IIIHH")
_rntuple_column_record_format = struct.Struct("<HHII")
_rntuple_alias_column = struct.Struct("<II")
_rntuple_extra_type_info = struct.Struct("<III")
_rntuple_record_size_format = struct.Struct("<I")
_rntuple_frame_header = struct.Struct("<ii")
_rntuple_cluster_group_format = struct.Struct("<I")
_rntuple_locator_format = struct.Struct("<iQ")
_rntuple_cluster_summary_format = struct.Struct("<QQ")


def from_zigzag(n):
    return n >> 1 ^ -(n & 1)


def _envelop_header(chunk, cursor, context):
    env_version, min_version = cursor.fields(
        chunk, uproot.const._rntuple_frame_format, context
    )
    return {"env_version": env_version, "min_version": min_version}


class Model_ROOT_3a3a_Experimental_3a3a_RNTuple(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::Experimental::RNTuple``.
    """

    @property
    def _keys(self):
        keys = []
        field_records = self.header.field_records
        for i, fr in enumerate(field_records):
            if fr.parent_field_id == i:
                keys.append(fr.field_name)
        return keys

    def keys(self):
        return self._keys

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

        (
            self._members["fCheckSum"],
            self._members["fVersion"],
            self._members["fSize"],
            self._members["fSeekHeader"],
            self._members["fNBytesHeader"],
            self._members["fLenHeader"],
            self._members["fSeekFooter"],
            self._members["fNBytesFooter"],
            self._members["fLenFooter"],
            self._members["fReserved"],
        ) = cursor.fields(chunk, _rntuple_format1, context)

        self._header_chunk_ready = False
        self._footer_chunk_ready = False
        self._header, self._footer = None, None

        self._field_names = None
        self._column_records = None

        self._page_list_envelopes = []

    def _prepare_header_chunk(self):
        context = {}
        seek, nbytes = self._members["fSeekHeader"], self._members["fNBytesHeader"]

        compressed_header_chunk = self.file.source.chunk(seek, seek + nbytes)

        if self._members["fNBytesHeader"] == self._members["fLenHeader"]:
            self._header_chunk = compressed_header_chunk
            self._header_cursor = uproot.source.cursor.Cursor(
                self._members["fSeekHeader"]
            )
        else:
            self._header_chunk = uproot.compression.decompress(
                compressed_header_chunk,
                uproot.source.cursor.Cursor(self._members["fSeekHeader"]),
                context,
                self._members["fNBytesHeader"],
                self._members["fLenHeader"],
            )
            self._header_cursor = uproot.source.cursor.Cursor(0)
        self._header_chunk_ready = True

    def _prepare_footer_chunk(self):
        context = {}
        seek, nbytes = self._members["fSeekFooter"], self._members["fNBytesFooter"]

        compressed_footer_chunk = self.file.source.chunk(seek, seek + nbytes)

        if self._members["fNBytesFooter"] == self._members["fLenFooter"]:
            self._footer_chunk = compressed_footer_chunk
            self._footer_cursor = uproot.source.cursor.Cursor(
                self._members["fSeekFooter"]
            )
        else:
            self._footer_chunk = uproot.compression.decompress(
                compressed_footer_chunk,
                uproot.source.cursor.Cursor(self._members["fSeekFooter"]),
                context,
                self._members["fNBytesFooter"],
                self._members["fLenFooter"],
            )
            self._footer_cursor = uproot.source.cursor.Cursor(0)
        self._footer_chunk_ready = True

    @property
    def header(self):
        if self._header is None:
            if not self._header_chunk_ready:
                self._prepare_header_chunk()
            context = {}
            cursor = self._header_cursor.copy()

            h = HeaderReader().read(self._header_chunk, cursor, context)
            self._header = h
            assert h.crc32 == zlib.crc32(self._header_chunk.raw_data[:-4])

        # cursor = self._header_cursor.copy()
        # cursor.debug(self._header_chunk)
        return self._header

    @property
    def field_names(self):
        if self._field_names is None:
            self._field_names = [r.field_name for r in self.header.field_records]
        return self._field_names

    @property
    def column_records(self):
        return self.header.column_records

    @property
    def footer(self):
        if self._footer is None:
            if not self._footer_chunk_ready:
                self._prepare_footer_chunk()
            cursor = self._footer_cursor.copy()
            context = {}

            f = FooterReader().read(self._footer_chunk, cursor, context)
            assert (
                f.header_crc32 == self.header.crc32
            ), f"crc32={self.header.crc32}, header_crc32={f.header_crc32}"
            assert f.crc32 == zlib.crc32(self._footer_chunk.raw_data[:-4])
            self._footer = f

        return self._footer

    @property
    def cluster_summaries(self):
        return self.footer.cluster_summaries

    # FIXME
    @property
    def _length(self):
        return sum(x.num_entries for x in self.cluster_summaries)

    def __len__(self):
        return self._length

    def read_locator(self, loc, uncomp_size, context):
        cursor = uproot.source.cursor.Cursor(loc.offset)
        chunk = self.file.source.chunk(loc.offset, loc.offset + loc.num_bytes)
        if loc.num_bytes < uncomp_size:
            decomp_chunk = uproot.compression.decompress(
                chunk, cursor, context, loc.num_bytes, uncomp_size, block_info=None
            )
            cursor.move_to(0)
        else:
            decomp_chunk = chunk
        return decomp_chunk, cursor

    @property
    def page_list_envelopes(self):
        context = {}

        if not self._page_list_envelopes:
            for record in self.footer.cluster_group_records:
                link = record.page_list_link
                loc = link.locator
                decomp_chunk, cursor = self.read_locator(
                    loc, link.env_uncomp_size, context
                )
                self._page_list_envelopes = PageLink().read(
                    decomp_chunk, cursor, context
                )

        return self._page_list_envelopes

    def base_col_form(self, cr, col_id, parameters=None):
        ak = uproot.extras.awkward()

        form_key = f"column-{col_id}"
        dtype_byte = cr.type
        if dtype_byte == uproot.const.rntuple_role_union:
            return form_key
        elif dtype_byte > uproot.const.rntuple_role_struct:
            dt_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
            if dt_str == "bit":
                dt_str = "bool"
            return ak.forms.NumpyForm(
                dt_str,
                form_key=form_key,
                parameters=parameters,
            )
        else:  # offset index column
            return form_key

    def col_form(self, field_id):
        ak = uproot.extras.awkward()

        # FIXME remove this ugly logic
        rel_crs = []
        rel_crs_idxs = []
        for i, cr in enumerate(self.header.column_records):
            if cr.field_id == field_id:
                rel_crs.append(cr)
                rel_crs_idxs.append(i)
            if cr.field_id > field_id:
                break
        if len(rel_crs) == 1:  # base case
            return self.base_col_form(rel_crs[0], rel_crs_idxs[0])
        elif (
            len(rel_crs_idxs) == 2
            and rel_crs[1].type == uproot.const.rntuple_col_type_to_num_dict["char"]
        ):
            # string field splits->2 in col records
            inner = self.base_col_form(
                rel_crs[1], rel_crs_idxs[-1], parameters={"__array__": "char"}
            )
            form_key = f"column-{rel_crs_idxs[0]}"
            return ak.forms.ListOffsetForm(
                "u32", inner, form_key=form_key, parameters={"__array__": "string"}
            )
        else:
            raise (RuntimeError(f"Missing special case: {field_id}"))

    def field_form(self, this_id, seen):
        ak = uproot.extras.awkward()

        field_records = self.header.field_records
        this_record = field_records[this_id]
        seen.append(this_id)
        structural_role = this_record.struct_role
        if (
            structural_role == uproot.const.rntuple_role_leaf
            and this_record.repetition == 0
        ):
            # base case of recursion
            # n.b. the split may happen in column
            return self.col_form(this_id)
        elif structural_role == uproot.const.rntuple_role_leaf:
            # std::array
            child_id = next(
                filter(
                    lambda i: field_records[i].parent_field_id == this_id,
                    range(this_id + 1, len(field_records)),
                )
            )
            inner = self.field_form(child_id, seen)
            return ak.forms.RegularForm(inner, this_record.repetition)
        elif structural_role == uproot.const.rntuple_role_vector:
            keyname = self.col_form(this_id)
            child_id = next(
                filter(
                    lambda i: field_records[i].parent_field_id == this_id,
                    range(this_id + 1, len(field_records)),
                )
            )
            inner = self.field_form(child_id, seen)
            return ak.forms.ListOffsetForm("u32", inner, form_key=keyname)
        elif structural_role == uproot.const.rntuple_role_struct:
            newids = []
            for i, fr in enumerate(field_records):
                if i not in seen and fr.parent_field_id == this_id:
                    newids.append(i)
            # go find N in the rest, N is the # of fields in struct
            recordlist = [self.field_form(i, seen) for i in newids]
            namelist = [field_records[i].field_name for i in newids]
            return ak.forms.RecordForm(recordlist, namelist, form_key="whatever")
        elif structural_role == uproot.const.rntuple_role_union:
            keyname = self.col_form(this_id)
            newids = []
            for i, fr in enumerate(field_records):
                if i not in seen and fr.parent_field_id == this_id:
                    newids.append(i)
            recordlist = [self.field_form(i, seen) for i in newids]
            return ak.forms.UnionForm("i8", "i64", recordlist, form_key=keyname)
        else:
            # everything should recurse above this branch
            raise AssertionError("this should be unreachable")

    def to_akform(self):
        ak = uproot.extras.awkward()

        field_records = self.header.field_records
        recordlist = []
        topnames = self.keys()
        seen = []
        for i in range(len(field_records)):
            if i not in seen:
                recordlist.append(self.field_form(i, seen))

        form = ak.forms.RecordForm(recordlist, topnames, form_key="toplevel")
        return form

    def read_pagedesc(self, destination, desc, dtype_str, dtype, nbits, split):
        loc = desc.locator
        context = {}
        # bool in RNTuple is always stored as bits
        isbit = dtype_str == "bit"
        len_divider = 8 if isbit else 1
        num_elements = len(destination)
        num_elements_toread = int(numpy.ceil(num_elements / len_divider))
        uncomp_size = num_elements_toread * dtype.itemsize
        decomp_chunk, cursor = self.read_locator(loc, uncomp_size, context)
        content = cursor.array(
            decomp_chunk, num_elements_toread, dtype, context, move=False
        )

        if split:
            content = content.view(numpy.uint8)

            if nbits == 16:
                # AAAAABBBBB needs to become
                # ABABABABAB
                res = numpy.empty(len(content), numpy.uint8)
                res[0::2] = content[len(res) * 0 // 2 : len(res) * 1 // 2]
                res[1::2] = content[len(res) * 1 // 2 : len(res) * 2 // 2]
                res = res.view(numpy.uint16)

            elif nbits == 32:
                # AAAAABBBBBCCCCCDDDDD needs to become
                # ABCDABCDABCDABCDABCD
                res = numpy.empty(len(content), numpy.uint8)
                res[0::4] = content[len(res) * 0 // 4 : len(res) * 1 // 4]
                res[1::4] = content[len(res) * 1 // 4 : len(res) * 2 // 4]
                res[2::4] = content[len(res) * 2 // 4 : len(res) * 3 // 4]
                res[3::4] = content[len(res) * 3 // 4 : len(res) * 4 // 4]
                res = res.view(numpy.uint32)

            elif nbits == 64:
                # AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH needs to become
                # ABCDEFGHABCDEFGHABCDEFGHABCDEFGHABCDEFGH
                res = numpy.empty(len(content), numpy.uint8)
                res[0::8] = content[len(res) * 0 // 8 : len(res) * 1 // 8]
                res[1::8] = content[len(res) * 1 // 8 : len(res) * 2 // 8]
                res[2::8] = content[len(res) * 2 // 8 : len(res) * 3 // 8]
                res[3::8] = content[len(res) * 3 // 8 : len(res) * 4 // 8]
                res[4::8] = content[len(res) * 4 // 8 : len(res) * 5 // 8]
                res[5::8] = content[len(res) * 5 // 8 : len(res) * 6 // 8]
                res[6::8] = content[len(res) * 6 // 8 : len(res) * 7 // 8]
                res[7::8] = content[len(res) * 7 // 8 : len(res) * 8 // 8]
                res = res.view(numpy.uint64)

            content = res

        if isbit:
            content = (
                numpy.unpackbits(content.view(dtype=numpy.uint8))
                .reshape(-1, 8)[:, ::-1]
                .reshape(-1)
            )

        # needed to chop off extra bits incase we used `unpackbits`
        destination[:] = content[:num_elements]

    def read_col_pages(self, ncol, cluster_range):
        return numpy.concatenate(
            [self.read_col_page(ncol, i) for i in cluster_range], axis=0
        )

    def read_col_page(self, ncol, cluster_i):
        linklist = self.page_list_envelopes.pagelinklist[cluster_i]
        pagelist = linklist[ncol]
        dtype_byte = self.column_records[ncol].type
        dtype_str = uproot.const.rntuple_col_num_to_dtype_dict[dtype_byte]
        dtype = numpy.dtype("bool") if dtype_str == "bit" else numpy.dtype(dtype_str)

        # FIXME vector read
        # n.b. it's possible pagelist is empty
        if not pagelist:
            return numpy.empty(0, dtype)
        total_len = numpy.sum([desc.num_elements for desc in pagelist])
        res = numpy.empty(total_len, dtype)
        tracker = 0
        split = 14 <= dtype_byte <= 21 or 26 <= dtype_byte <= 28
        nbits = uproot.const.rntuple_col_num_to_size_dict[dtype_byte]
        for page_desc in pagelist:
            n_elements = page_desc.num_elements
            tracker_end = tracker + n_elements
            self.read_pagedesc(
                res[tracker:tracker_end], page_desc, dtype_str, dtype, nbits, split
            )
            tracker = tracker_end

        if dtype_byte <= uproot.const.rntuple_col_type_to_num_dict["index32"]:
            res = numpy.insert(res, 0, 0)  # for offsets
        zigzag = 26 <= dtype_byte <= 28
        delta = 14 <= dtype_byte <= 15
        if zigzag:
            res = from_zigzag(res)
        elif delta:
            numpy.cumsum(res)
        return res

    def arrays(
        self,
        filter_names="*",
        filter_typenames=None,
        entry_start=0,
        entry_stop=None,
        decompression_executor=None,
        array_cache=None,
    ):
        ak = uproot.extras.awkward()

        entry_stop = entry_stop or self._length

        clusters = self.cluster_summaries
        cluster_starts = numpy.array([c.num_first_entry for c in clusters])

        start_cluster_idx = (
            numpy.searchsorted(cluster_starts, entry_start, side="right") - 1
        )
        stop_cluster_idx = numpy.searchsorted(cluster_starts, entry_stop, side="right")
        cluster_num_entries = numpy.sum(
            [c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]]
        )

        form = self.to_akform().select_columns(filter_names)
        # only read columns mentioned in the awkward form
        target_cols = []
        container_dict = {}
        _recursive_find(form, target_cols)
        for i, cr in enumerate(self.column_records):
            key = f"column-{i}"
            dtype_byte = cr.type
            if key in target_cols:
                content = self.read_col_pages(
                    i, range(start_cluster_idx, stop_cluster_idx)
                )
                if dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:
                    kindex, tags = _split_switch_bits(content)
                    container_dict[f"{key}-index"] = kindex
                    container_dict[f"{key}-tags"] = tags
                else:
                    # don't distinguish data and offsets
                    container_dict[f"{key}-data"] = content
                    container_dict[f"{key}-offsets"] = content
        cluster_offset = cluster_starts[start_cluster_idx]
        entry_start -= cluster_offset
        entry_stop -= cluster_offset
        return ak.from_buffers(form, cluster_num_entries, container_dict)[
            entry_start:entry_stop
        ]


# Supporting function and classes
def _split_switch_bits(content):
    kindex = numpy.bitwise_and(content, numpy.int64(0x00000000000FFFFF))
    tags = (content >> 44).astype("int8") - 1
    return kindex, tags


def _recursive_find(form, res):
    ak = uproot.extras.awkward()

    if hasattr(form, "form_key"):
        res.append(form.form_key)
    if hasattr(form, "contents"):
        for c in form.contents:
            _recursive_find(c, res)
    if hasattr(form, "content") and issubclass(type(form.content), ak.forms.Form):
        _recursive_find(form.content, res)


class PageDescription:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.num_elements = cursor.field(chunk, struct.Struct("<I"), context)
        out.locator = LocatorReader().read(chunk, cursor, context)
        return out


class PageLinkInner:
    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes, num_pages = local_cursor.fields(
            chunk, _rntuple_frame_header, context
        )
        assert num_bytes < 0, f"num_bytes={num_bytes}"
        cursor.skip(-num_bytes)
        return [
            PageDescription().read(chunk, local_cursor, context)
            for _ in range(num_pages)
        ]


class PageLink:
    def __init__(self):
        self.top_most_list = ListFrameReader(  # top-most list
            ListFrameReader(PageLinkInner())  # outer list (inner list)
        )

    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.env_header = _envelop_header(chunk, cursor, context)
        local_cursor = cursor.copy()
        num_bytes, num_items = cursor.fields(chunk, _rntuple_frame_header, context)
        if num_items == 0:
            return out
        out.pagelinklist = self.top_most_list.read(chunk, local_cursor, context)
        cursor.skip(-num_bytes - 8)
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)
        assert zlib.crc32(chunk.raw_data[:-4]) == out.crc32
        return out


class LocatorReader:
    def read(self, chunk, cursor, context):
        out = MetaData("Locator")
        out.num_bytes, out.offset = cursor.fields(
            chunk, _rntuple_locator_format, context
        )
        return out


class EnvLinkReader:
    def read(self, chunk, cursor, context):
        out = MetaData("EnvLink")
        out.env_uncomp_size = cursor.field(chunk, struct.Struct("<I"), context)
        out.locator = LocatorReader().read(chunk, cursor, context)
        return out


class MetaData:
    def __init__(self, name, **kwargs):
        self.__dict__["_name"] = name
        self.__dict__["_fields"] = kwargs

    @property
    def name(self):
        return self.__dict__["_name"]

    def __repr__(self):
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__["_fields"].items())
        return f"MetaData({self.name!r}, {kwargs})"

    def __getattr__(self, name):
        if not name.startswith("_"):
            return self.__dict__["_fields"][name]
        else:
            return self.__dict__[name]

    def __setattr__(self, name, val):
        self.__dict__["_fields"][name] = val


class RecordFrameReader:
    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes = local_cursor.field(chunk, _rntuple_record_size_format, context)
        cursor.skip(num_bytes)
        return self.payload.read(chunk, local_cursor, context)


class ListFrameReader:
    def __init__(self, payload):
        self.payload = payload

    def read(self, chunk, cursor, context):
        local_cursor = cursor.copy()
        num_bytes, num_items = local_cursor.fields(
            chunk, _rntuple_frame_header, context
        )
        assert num_bytes < 0, f"num_bytes={num_bytes}"
        cursor.skip(-num_bytes)
        return [
            self.payload.read(chunk, local_cursor, context) for _ in range(num_items)
        ]


# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#field-description
class FieldRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("FieldRecordFrame")
        (
            out.field_version,
            out.type_version,
            out.parent_field_id,
            out.struct_role,
            out.flags,
        ) = cursor.fields(chunk, _rntuple_field_description, context)
        if out.flags == 0x0001:
            out.repetition = cursor.field(chunk, struct.Struct("Q"), context)
        else:
            out.repetition = 0

        out.field_name, out.type_name, out.type_alias, out.field_desc = (
            cursor.rntuple_string(chunk, context) for i in range(4)
        )
        return out


# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#column-description
class ColumnRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ColumnRecordFrame")
        out.type, out.nbits, out.field_id, out.flags = cursor.fields(
            chunk, _rntuple_column_record_format, context
        )
        return out


class AliasColumnReader:
    def read(self, chunk, cursor, context):
        out = MetaData("AliasColumn")

        out.physical_id, out.field_id = cursor.fields(
            chunk, _rntuple_alias_column, context
        )
        return out


class ExtraTypeInfoReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ExtraTypeInfoReader")

        out.type_ver_from, out.type_ver_to, out.content_id = cursor.fields(
            chunk, _rntuple_extra_type_info, context
        )
        out.type_name = cursor.rntuple_string(chunk, context)
        return out


class HeaderReader:
    def __init__(self):
        self.list_field_record_frames = ListFrameReader(
            RecordFrameReader(FieldRecordReader())
        )
        self.list_column_record_frames = ListFrameReader(
            RecordFrameReader(ColumnRecordReader())
        )
        self.list_alias_column_frames = ListFrameReader(
            RecordFrameReader(AliasColumnReader())
        )
        self.list_extra_type_info_reader = ListFrameReader(
            RecordFrameReader(ExtraTypeInfoReader())
        )

    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.env_header = _envelop_header(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.rc_tag = cursor.field(chunk, struct.Struct("I"), context)
        out.name, out.ntuple_description, out.writer_identifier = (
            cursor.rntuple_string(chunk, context) for _ in range(3)
        )

        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_columns = self.list_alias_column_frames.read(chunk, cursor, context)
        out.extra_type_infos = self.list_extra_type_info_reader.read(
            chunk, cursor, context
        )
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)

        return out

    def read_extension_header(self, out, chunk, cursor, context):
        out.field_records = self.list_field_record_frames.read(chunk, cursor, context)
        out.column_records = self.list_column_record_frames.read(chunk, cursor, context)
        out.alias_columns = self.list_alias_column_frames.read(chunk, cursor, context)
        out.extra_type_infos = self.list_extra_type_info_reader.read(
            chunk, cursor, context
        )
        return out


class ColumnGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, self._cluster_summary_format, context
        )
        return out


class ClusterSummaryReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterSummaryRecord")
        out.num_first_entry, out.num_entries = cursor.fields(
            chunk, _rntuple_cluster_summary_format, context
        )
        return out


class ClusterGroupRecordReader:
    def read(self, chunk, cursor, context):
        out = MetaData("ClusterGroupRecord")
        out.num_clusters = cursor.field(chunk, _rntuple_cluster_group_format, context)
        out.page_list_link = EnvLinkReader().read(chunk, cursor, context)
        return out


class RNTupleSchemaExtension:
    def read(self, chunk, cursor, context):
        out = MetaData(type(self).__name__)
        out.size = cursor.field(chunk, struct.Struct("<I"), context)
        out.field_records = ListFrameReader(
            RecordFrameReader(FieldRecordReader())
        ).read(chunk, cursor, context)
        out.column_records = ListFrameReader(
            RecordFrameReader(ColumnRecordReader())
        ).read(chunk, cursor, context)
        out.alias_records = ListFrameReader(
            RecordFrameReader(AliasColumnReader())
        ).read(chunk, cursor, context)
        out.extra_type_info = ListFrameReader(
            RecordFrameReader(ExtraTypeInfoReader())
        ).read(chunk, cursor, context)
        return out


class FooterReader:
    def __init__(self):
        self.extension_header_links = RNTupleSchemaExtension()
        # self.extension_header_links = ListFrameReader(EnvLinkReader())
        self.column_group_record_frames = ListFrameReader(
            RecordFrameReader(ColumnGroupRecordReader())
        )
        self.cluster_summary_frames = ListFrameReader(
            RecordFrameReader(ClusterSummaryReader())
        )
        self.cluster_group_record_frames = ListFrameReader(
            RecordFrameReader(ClusterGroupRecordReader())
        )
        self.meta_data_links = ListFrameReader(EnvLinkReader())

    def read(self, chunk, cursor, context):
        out = MetaData("Footer")
        out.env_header = _envelop_header(chunk, cursor, context)
        out.feature_flag = cursor.field(chunk, _rntuple_feature_flag_format, context)
        out.header_crc32 = cursor.field(chunk, struct.Struct("<I"), context)
        out.extension_links = self.extension_header_links.read(chunk, cursor, context)

        out.col_group_records = self.column_group_record_frames.read(
            chunk, cursor, context
        )
        out.cluster_summaries = self.cluster_summary_frames.read(chunk, cursor, context)
        out.cluster_group_records = self.cluster_group_record_frames.read(
            chunk, cursor, context
        )
        out.meta_block_links = self.meta_data_links.read(chunk, cursor, context)
        out.crc32 = cursor.field(chunk, struct.Struct("<I"), context)
        return out


uproot.classes[
    "ROOT::Experimental::RNTuple"
] = Model_ROOT_3a3a_Experimental_3a3a_RNTuple
