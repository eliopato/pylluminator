# Most of this code comes from https://github.com/FoxoTech/methylprep

from enum import IntEnum, unique
import pandas as pd
import numpy as np
import struct
from pathlib import PurePath
import gzip
import logging

DEFAULT_IDAT_VERSION = 3
DEFAULT_IDAT_FILE_ID = 'IDAT'


LOGGER = logging.getLogger(__name__)


def npread(file_like, dtype: str, n: int) -> np.ndarray:
    """Parses a binary file multiple times, allowing for control if the file ends prematurely. This replaces
     read_results() and runs faster, and it provides support for reading gzipped idat files without decompressing.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.
        dtype {data type} -- used within idat files, 2-bit, or 4-bit numbers stored in binary at specific addresses
        n {number of snps read} -- see files/idat.py for how this function is applied.

    Raises:
        EOFError: If the end of the file is reached before the number of elements have
            been processed.

    Returns:
        A list of the parsed values.
    """
    dtype = np.dtype(dtype)
    # np.readfile is not able to read from gzopene-d file
    alldata = file_like.read(dtype.itemsize * n)
    if len(alldata) != dtype.itemsize * n:
        raise EOFError('End of file reached before number of results parsed')
    read_data = np.frombuffer(alldata, dtype, n)
    if read_data.size != n:
        raise EOFError('End of file reached before number of results parsed')
    return read_data


def read_byte(infile) -> int:
    """Converts a single byte to an integer value.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.

    Returns:
        [integer] -- Unsigned integer value converted from the supplied bytes.
    """
    return bytes_to_int(infile.read(1), signed=False)


def bytes_to_int(input_bytes, signed=False):
    """Returns the integer represented by the given array of bytes. Pre-sets the byteorder to be little-endian.

    Arguments:
        input_bytes -- Holds the array of bytes to convert. The argument must either support the buffer protocol or
            be an iterable object producing bytes. Bytes and bytearray are examples of built-in objects that support the
            buffer protocol.

    Keyword Arguments:
        signed {bool} -- Indicates whether two's complement is used to represent the integer. (default: {False})

    Returns:
        [integer] -- Integer value converted from the supplied bytes.
    """
    return int.from_bytes(input_bytes, byteorder='little', signed=signed)


def read_string(infile) -> str:
    """Converts an array of bytes to a string.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.

    Returns:
        [string] -- UTF-8 decoded string value.
    """
    num_bytes = read_byte(infile)
    num_chars = num_bytes % 128
    shift = 0

    while num_bytes // 128 == 1:
        num_bytes = read_byte(infile)
        shift += 7
        offset = (num_bytes % 128) * (2 ** shift)
        num_chars += offset

    return read_char(infile, num_chars)


def read_short(infile) -> int:
    """Converts a two-byte element to an integer value.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.

    Returns:
        [integer] -- Unsigned integer value converted from the supplied bytes.
    """
    return bytes_to_int(infile.read(2), signed=False)


def read_int(infile) -> int:
    """Converts a four-byte element to an integer value.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.

    Returns:
        [integer] -- Signed integer value converted from the supplied bytes.
    """
    return bytes_to_int(infile.read(4), signed=True)


def read_long(infile) -> int:
    """Converts an eight-byte element to an integer value.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.

    Returns:
        [integer] -- Signed integer value converted from the supplied bytes.
    """
    return bytes_to_int(infile.read(8), signed=True)


def read_char(infile, num_bytes: int) -> str:
    """Converts an array of bytes to a string.

    Arguments:
        infile {file-like} -- The binary file to read the select number of bytes.
        num_bytes {integer} -- The number of bytes to read and parse.

    Returns:
        [string] -- UTF-8 decoded string value.
    """
    return infile.read(num_bytes).decode('utf-8')


def read_and_reset(inner):
    """Decorator that resets a file-like object back to the original position after the function has been called."""

    def wrapper(infile, *args, **kwargs):
        current_position = infile.tell()
        r_val = inner(infile, *args, **kwargs)
        infile.seek(current_position)
        return r_val

    return wrapper


def get_file_object(filepath):
    """Returns a file-like object based on the provided input. If the input argument is a string, it will attempt to
    open the file in 'rb' mode."""
    if pd.api.types.is_file_like(filepath):
        return filepath

    if PurePath(filepath).suffix == '.gz':
        return gzip.open(filepath, 'rb')

    return open(filepath, 'rb')


@unique
class IdatHeaderLocation(IntEnum):
    """Unique IntEnum defining constant values for byte offsets of IDAT headers."""
    FILE_TYPE = 0
    VERSION = 4
    FIELD_COUNT = 12
    SECTION_OFFSETS = 16


@unique
class IdatSectionCode(IntEnum):
    """Unique IntEnum defining constant values for byte offsets of IDAT headers.
    These values come from the field codes of the Bioconductor illuminaio package.

    MM: refer to https://bioconductor.org/packages/release/bioc/vignettes/illuminaio/inst/doc/EncryptedFormat.pdf
    and https://bioconductor.org/packages/release/bioc/vignettes/illuminaio/inst/doc/illuminaio.pdf
    and source: https://github.com/snewhouse/glu-genetics/blob/master/glu/lib/illumina.py
    """
    ILLUMINA_ID = 102
    STD_DEV = 103
    MEAN = 104
    NUM_BEADS = 107  # how many replicate measurements for each probe
    MID_BLOCK = 200
    RUN_INFO = 300
    RED_GREEN = 400
    MOSTLY_NULL = 401  # manifest
    BARCODE = 402
    CHIP_TYPE = 403  # format
    MOSTLY_A = 404  # label
    OPA = 405
    SAMPLE_ID = 406
    DESCR = 407
    PLATE = 408
    WELL = 409
    UNKNOWN_6 = 410
    UNKNOWN_7 = 510
    NUM_SNPS_READ = 1000


class IdatDataset:
    """Validates and parses an Illumina IDAT file.

    Arguments:
        filepath {file-like} -- the IDAT file to parse.
        get_std_dev {bool} --
        get_n_beads {bool} --

    Keyword Arguments:
        bit {default 'float32'} -- 'float16' will pre-normalize intensities, capping max intensity at 32127. This cuts
            data size in half, but will reduce precision on ~0.01% of probes. [effectively downscaling fluorescence]

    Raises:
        ValueError: The IDAT file has an incorrect identifier or version specifier.
    """

    def __init__(self, filepath: str, bit='float32'):
        """Initializes the IdatDataset, reads and parses the IDAT file."""
        self.barcode = None
        self.chip_type = None
        self.n_snps_read = 0
        self.run_info = []
        self.bit = bit

        with get_file_object(filepath) as idat_file:
            # assert file is indeed IDAT format
            if not self.is_idat_file(idat_file, DEFAULT_IDAT_FILE_ID):
                raise ValueError('Not an IDAT file. Unsupported file type.')

            # assert correct IDAT file version
            if not self.is_correct_version(idat_file, DEFAULT_IDAT_VERSION):
                raise ValueError('Not a version 3 IDAT file. Unsupported IDAT version.')

            self.probe_means = self.read(idat_file)
            if self.overflow_check() is False:
                LOGGER.warning("IDAT: contains negative probe values (uint16 overflow error)")

    @staticmethod
    @read_and_reset
    def is_idat_file(idat_file, expected) -> bool:
        """Checks if the provided file has the correct identifier.

        Arguments:
            idat_file {file-like} -- the IDAT file to check.
            expected {string} -- expected IDAT file identifier.

        Returns:
            [boolean] -- If the IDAT file identifier matches the expected value
        """
        idat_file.seek(IdatHeaderLocation.FILE_TYPE.value)
        file_type = read_char(idat_file, len(expected))
        return file_type.lower() == expected.lower()

    @staticmethod
    @read_and_reset
    def is_correct_version(idat_file, expected: int) -> bool:
        """Checks if the provided file has the correct version.

        Arguments:
            idat_file {file-like} -- the IDAT file to check.
            expected {integer} -- expected IDAT version.

        Returns:
            [boolean] -- If the IDAT file version matches the expected value
        """
        idat_file.seek(IdatHeaderLocation.VERSION.value)
        idat_version = read_long(idat_file)
        return str(idat_version) == str(expected)

    @staticmethod
    @read_and_reset
    def get_section_offsets(idat_file) -> dict:
        """Parses the IDAT file header to get the byte position for the start of each section.

        Arguments:
            idat_file {file-like} -- the IDAT file to process.

        Returns:
            [dict] -- The byte offset for each file section.
        """
        idat_file.seek(IdatHeaderLocation.FIELD_COUNT.value)
        num_fields = read_int(idat_file)

        idat_file.seek(IdatHeaderLocation.SECTION_OFFSETS.value)

        offsets = {}
        for _idx in range(num_fields):
            key = read_short(idat_file)
            offsets[key] = read_long(idat_file)

        return offsets

    def read(self, idat_file) -> pd.DataFrame:
        """Reads the IDAT file and parses the appropriate sections. Joins the mean probe intensity values with their
        Illumina probe ID.

        Arguments:
            idat_file {file-like} -- the IDAT file to process.

        Returns:
            DataFrame -- mean probe intensity values indexed by Illumina ID.
        """
        section_offsets = self.get_section_offsets(idat_file)

        def seek_to_section(section_code):
            offset = section_offsets[section_code.value]
            idat_file.seek(offset)

        seek_to_section(IdatSectionCode.BARCODE)
        self.barcode = read_string(idat_file)

        seek_to_section(IdatSectionCode.CHIP_TYPE)
        self.chip_type = read_string(idat_file)

        seek_to_section(IdatSectionCode.NUM_SNPS_READ)
        self.n_snps_read = read_int(idat_file)

        seek_to_section(IdatSectionCode.ILLUMINA_ID)
        illumina_ids = npread(idat_file, '<i4', self.n_snps_read)

        seek_to_section(IdatSectionCode.MEAN)
        probe_means = npread(idat_file, '<u2', self.n_snps_read)  # '<u2' reads data as numpy unsigned-float16

        seek_to_section(IdatSectionCode.RUN_INFO)
        run_info_entry_count, = struct.unpack('<L', idat_file.read(4))
        for i in range(run_info_entry_count):
            timestamp = read_string(idat_file)
            entry_type = read_string(idat_file)
            parameters = read_string(idat_file)
            codeblock = read_string(idat_file)
            code_version = read_string(idat_file)
            self.run_info.append((timestamp, entry_type, parameters, codeblock, code_version))

        data = {'mean_value': probe_means}

        seek_to_section(IdatSectionCode.STD_DEV)
        data['std_dev'] = npread(idat_file, '<u2', self.n_snps_read)

        seek_to_section(IdatSectionCode.NUM_BEADS)
        data['n_beads'] = npread(idat_file, '<u1', self.n_snps_read)

        data_frame = pd.DataFrame(data=data, index=illumina_ids, dtype=self.bit)
        data_frame.index.name = 'illumina_id'

        if self.bit == 'float16':
            data_frame = data_frame.clip(upper=32127)
            data_frame = data_frame.astype('int16')

        return data_frame

    def overflow_check(self) -> bool:
        if hasattr(self, 'probe_means'):
            if (self.probe_means.values < 0).any():
                return False
        return True

    def __str__(self):
        return f'{self.probe_means.head(3)}'
