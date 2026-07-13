from services.ltx_pipeline_common import default_tiling_config


def test_default_tiling_config_uses_quality_overlap() -> None:
    config = default_tiling_config()
    assert config.spatial_config is not None
    assert config.temporal_config is not None
    assert config.spatial_config.tile_size_in_pixels == 768
    assert config.spatial_config.tile_overlap_in_pixels == 256
    assert config.temporal_config.tile_size_in_frames == 80
    assert config.temporal_config.tile_overlap_in_frames == 24
