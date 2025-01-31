from pylluminator.annotations import get_or_download_annotation_data, Channel, GenomeVersion, GenomeInfo, Annotations, \
    ArrayType, detect_array
import pytest

def test_failed_download():
    assert get_or_download_annotation_data('bla', 'type', 'data', 'https://www.fakeurl.co/fakefile') is None

def test_channel():
    red_channel = Channel('Red')
    assert red_channel.is_red is True
    assert red_channel.is_green is False
    green_channel = Channel('Grn')
    assert green_channel.is_red is False
    assert green_channel.is_green is True
    with pytest.raises(ValueError):
        Channel('wronginput')

def test_genome_version():
    gv = GenomeVersion('hg38')
    assert gv.is_human()
    gv = GenomeVersion('hg19')
    assert gv.is_human()
    gv = GenomeVersion('mm10')
    assert gv.is_human() is False
    gv = GenomeVersion('mm39')
    assert gv.is_human() is False

def test_genome_info():
    with pytest.raises(ValueError):
        GenomeInfo('name', None)
    with pytest.raises(ValueError):
        GenomeInfo('name', GenomeVersion('hg19'))
    GenomeInfo('default', GenomeVersion('hg19'))
    GenomeInfo('default', GenomeVersion('hg38'))
    GenomeInfo('default', GenomeVersion('mm10'))
    GenomeInfo('default', GenomeVersion('mm39'))

def test_failed_annotation():
    with pytest.raises(ValueError):
        Annotations(ArrayType.HUMAN_EPIC_V2, GenomeVersion.HG38, 'name')

def test_failed_genomic_ranges():
    anno = Annotations(ArrayType.HUMAN_EPIC_V2, GenomeVersion.HG38)
    anno.probe_infos = None
    assert anno.make_genomic_ranges() is None

def test_failed_array_detection():
    assert detect_array(3000) is ArrayType.HUMAN_EPIC_V2