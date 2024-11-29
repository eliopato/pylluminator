"""
Functions to parse .idat files.

Most of this code comes from https://github.com/FoxoTech/methylprep
"""


from enum import IntEnum, unique
import pandas as pd
import numpy as np
import struct
from pathlib import PurePath
import gzip
import os

from pylluminator.utils import get_logger

LOGGER = get_logger()

DEFAULT_IDAT_VERSION = 3
DEFAULT_IDAT_FILE_ID = 'IDAT'


def npread(file_like, dtype: str, n: int) -> np.ndarray:
    """Parses a binary file multiple times, allowing for control if the file ends prematurely. This replaces
     read_results() and runs faster, and it provides support for reading gzipped idat files without decompressing.

    :param file_like: The binary file to read the select number of bytes.
    :type file_like: file-like
    :param dtype: used within idat files, 2-bit, or 4-bit numbers stored in binary at specific addresses
    :type dtype: str
    :param n: number of snps read
    :type n: int

    :raises: EOFError: If the end of the file is reached before the number of elements have been processed.

    :return: A list of the parsed values.
    :rtype: numpy.ndarray
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

    :param infile: The binary file to read the select number of bytes.
    :type infile: file-like

    :return: Unsigned integer value converted from the supplied bytes.
    :rtype: int
    """
    return bytes_to_int(infile.read(1), signed=False)


def bytes_to_int(input_bytes, signed=False) -> int:
    """Returns the integer represented by the given array of bytes. Pre-sets the byteorder to be little-endian.

    :param input_bytes: Holds the array of bytes to convert. The argument must either support the buffer protocol or
        be an iterable object producing bytes. Bytes and bytearray are examples of built-in objects that support the
        buffer protocol.
    :param signed: Indicates whether two's complement is used to represent the integer - Default False
    :type signed: bool

    :return: Integer value converted from the supplied bytes.
    :rtype: int"""
    return int.from_bytes(input_bytes, byteorder='little', signed=signed)


def read_string(infile) -> str:
    """Converts an array of bytes to a string.

    :param infile: The binary file to read the select number of bytes.
    :type infile: file-like

    :return: UTF-8 decoded string value.
    :rtype: str
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

    :param infile: The binary file to read the select number of bytes.
    :type infile: file-like

    :return: Unsigned integer value converted from the supplied bytes.
    :rtype: int"""
    return bytes_to_int(infile.read(2), signed=False)


def read_int(infile) -> int:
    """Converts a four-byte element to an integer value.

    :param infile: The binary file to read the select number of bytes.
    :type infile: file-like

    :return: Signed integer value converted from the supplied bytes.
    :rtype: int"""
    return bytes_to_int(infile.read(4), signed=True)


def read_long(infile) -> int:
    """Converts an eight-byte element to an integer value.

    :param infile: The binary file to read the select number of bytes.
    :type infile: file-like

    :return: Signed integer value converted from the supplied bytes.
    :rtype: int"""
    return bytes_to_int(infile.read(8), signed=True)


def read_char(infile, num_bytes: int) -> str:
    """Converts an array of bytes to a string.

    :param infile: The binary file to read the select number of bytes.
    :type infile: file-like
    :param num_bytes: The number of bytes to read and parse.
    :type num_bytes: int

    :return: UTF-8 decoded string value.
    :rtype: str"""
    return infile.read(num_bytes).decode('utf-8')


def read_and_reset(inner):
    """Decorator that resets a file-like object back to the original position after the function has been called.

    :param inner: file to read and reset
    :type inner: file-like"""

    def wrapper(infile, *args, **kwargs):
        current_position = infile.tell()
        r_val = inner(infile, *args, **kwargs)
        infile.seek(current_position)
        return r_val

    return wrapper


def get_file_object(filepath):
    """Get an opened file object. Unzip if filepath is a .gz file.

    :param filepath: a string, or path-like object. If the input argument is a string, it will attempt to
        open the file in 'rb' mode

    :return: a file-like object based on the provided input."""
    filepath = os.path.expanduser(filepath)

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

    :ivar barcode:
    :vartype barcode:

    :ivar chip_type:
    :vartype chip_type:

    :ivar n_snps_read. Default: 0
    :vartype n_snps_read:

    :ivar run_info: Default []
    :vartype run_info:

    :ivar bit: Defines the data type, hence the precision. Default: 'float32'
    :vartype bit: str

    :ivar probes_df: dataframe with the .idat file data parsed (illumina IDs, mean_value, std_dev, n_beads)
    :vartype probes_df: pandas.DataFrame
    """

    def __init__(self, filepath: str, bit='float32'):
        """Initializes the IdatDataset by reading the .idat file provided.

        :param filepath: the IDAT file to parse.
        :type filepath: str
        :param bit: Either 'float32' or 'float16'. 'float16' will pre-normalize intensities, capping max intensity at 32127.
            This cuts data size in half, but will reduce precision on ~0.01% of probes. [effectively downscaling fluorescence]
            Default: 'float32'
        :type bit: str

        :raises: ValueError: The IDAT file has an incorrect identifier or version specifier."""

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

            self.probes_df = self.read(idat_file)
            if self.overflow_check() is False:
                LOGGER.warning("IDAT: contains negative probe values (uint16 overflow error)")

    @staticmethod
    @read_and_reset
    def is_idat_file(idat_file, expected) -> bool:
        """Checks if the provided file has the correct identifier.

        :param idat_file: the IDAT file to check.
        :type idat_file: file-like
        :param expected: expected IDAT file identifier.
        :type expected: str

        :return: If the IDAT file identifier matches the expected value
        :rtype: bool"""
        idat_file.seek(IdatHeaderLocation.FILE_TYPE.value)
        file_type = read_char(idat_file, len(expected))
        return file_type.lower() == expected.lower()

    @staticmethod
    @read_and_reset
    def is_correct_version(idat_file, expected: int) -> bool:
        """Checks if the provided file has the correct version.

        :param idat_file: the IDAT file to check.
        :type idat_file: file-like
        :param expected: expected IDAT version.
        :type expected: int

        :return: If the IDAT file version matches the expected value
        :rtype: bool"""
        idat_file.seek(IdatHeaderLocation.VERSION.value)
        idat_version = read_long(idat_file)
        return str(idat_version) == str(expected)

    @staticmethod
    @read_and_reset
    def get_section_offsets(idat_file) -> dict:
        """Parses the IDAT file header to get the byte position for the start of each section.

        :param idat_file: the IDAT file to process.
        :type idat_file: file-like

        :return: The byte offset for each file section.
        :rtype: dict"""
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

        :param idat_file: the IDAT file to process.
        :type idat_file: file-like

        :return: mean probe intensity values indexed by Illumina ID.
        :rtype: pandas.DataFrame"""
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
        probes_df = npread(idat_file, '<u2', self.n_snps_read)  # '<u2' reads data as numpy unsigned-float16

        seek_to_section(IdatSectionCode.RUN_INFO)
        run_info_entry_count, = struct.unpack('<L', idat_file.read(4))
        for i in range(run_info_entry_count):
            timestamp = read_string(idat_file)
            entry_type = read_string(idat_file)
            parameters = read_string(idat_file)
            codeblock = read_string(idat_file)
            code_version = read_string(idat_file)
            self.run_info.append((timestamp, entry_type, parameters, codeblock, code_version))

        data = {'mean_value': probes_df}

        seek_to_section(IdatSectionCode.STD_DEV)
        data['std_dev'] = npread(idat_file, '<u2', self.n_snps_read)

        seek_to_section(IdatSectionCode.NUM_BEADS)
        data['n_beads'] = npread(idat_file, '<u1', self.n_snps_read)

        data_frame = pd.DataFrame(data=data, index=illumina_ids, dtype=self.bit)
        data_frame.index.name = 'illumina_id'

        if self.bit == 'float16':
            data_frame = data_frame.clip(upper=32127)
            data_frame = data_frame.astype('int16')
        # elif self.bit == 'float32':
        #     data_frame = data_frame.clip(upper=2147483647)
        #     data_frame = data_frame.astype('int32')

        return data_frame

    def overflow_check(self) -> bool:
        """Check if there is any negative value in the dataframe, meaning there was an overflow

        :return: True if an overflow was detected in any value
        :rtype: bool"""
        if hasattr(self, 'probes_df'):
            if (self.probes_df.values < 0).any():
                return False
        return True

    def __str__(self):
        return f'IdatDataset object.\nHead of self.probes_df dataframe : \n{self.probes_df.head(3)}'

    def __repr__(self):
        return self.__str__()