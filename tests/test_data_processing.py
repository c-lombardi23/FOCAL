"""
Tests for the data processing module.
"""

import pytest
import pandas as pd
from cleave_app.data_processing import DataCollector
from io import BytesIO
from PIL import Image
import numpy as np


@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    fake_image = BytesIO()
    image = Image.new("RGB", (100, 100))
    image.save(fake_image, format="PNG")
    fake_image.seek(0)
    return fake_image


def test_data_collector_full(tmp_path, mock_image):
    """Test the DataCollector class with a complete dataset."""
    # Create test directory structure
    img_folder = tmp_path / "images"
    img_folder.mkdir()
    image_path = img_folder / "image1.png"
    with open(image_path, "wb") as f:
        f.write(mock_image.read())

    # Create test CSV data
    csv_content = """ImagePath,FiberType,DateCreated,Diameter,CleaveAngle,CleaveTension,TensionVelocity,FHBOffset,ScribeDiameter,Misting,Hackle,Tearing
image1.png,PM15U25d,2025-06-09 15:37,123.5,0.22,193,60,2552,17.28,0,0,1
"""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(csv_content)

    # Test DataCollector initialization
    collector = DataCollector(csv_path=str(csv_path), img_folder=str(img_folder) + "/")

    # Verify the dataframe was created correctly
    assert collector.df is not None
    assert 'CleaveCategory' in collector.df.columns
    assert isinstance(collector.df['ImagePath'][0], str)
    assert collector.df['ImagePath'][0] == "image1.png"
    
    # Test data extraction
    images, features, labels = collector.extract_data()
    assert len(images) == 1
    assert features.shape == (1, 6)  # 6 features
    assert labels.shape == (1, 5)    # 5 classes


def test_data_collector_invalid_paths():
    """Test DataCollector with invalid paths."""
    with pytest.raises(ValueError, match="Must provide data path"):
        DataCollector(None, None)
    
    with pytest.raises(ValueError, match="Must provide data path"):
        DataCollector("", "")


def test_data_collector_file_not_found(tmp_path):
    """Test DataCollector with non-existent files."""
    non_existent_csv = tmp_path / "nonexistent.csv"
    non_existent_folder = tmp_path / "nonexistent"
    
    with pytest.raises(ValueError, match="does not exist"):
        DataCollector(str(non_existent_csv), str(non_existent_folder))
