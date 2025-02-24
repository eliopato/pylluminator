from pylluminator.quality_control import (detection_stats, intensity_stats, nb_probes_stats, type1_color_channels_stats,
                                          dye_bias_stats, betas_stats)

def test_one_sample(test_samples):
    sample_name = 'LNCAP_500_3'
    detection_stats(test_samples, sample_name)
    intensity_stats(test_samples, sample_name)
    nb_probes_stats(test_samples, sample_name)
    type1_color_channels_stats(test_samples, sample_name)
    dye_bias_stats(test_samples, sample_name)
    betas_stats(test_samples, sample_name)
