"""
Unit Tests
==========
Test your data processing and modeling functions here.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing import clean_dataframe


class TestDataProcessing:
    """Tests for data processing functions."""
    
    def test_clean_dataframe_removes_duplicates(self):
        """Test that duplicates are removed."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = clean_dataframe(df)
        assert len(result) == 2
    
    def test_clean_dataframe_returns_dataframe(self):
        """Test that output is a DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = clean_dataframe(df)
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
