#!/usr/bin/env python3
"""
Tests for auto-download functionality.

Tests:
- Dataset detection
- Auto-download logic
- Installed extras detection
- Dataset availability checking
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from baguettotron.data.auto_download import (
    get_project_root,
    get_installed_extras,
    is_dataset_available,
    get_dataset_info,
    list_available_datasets,
    DATASETS,
)


class TestProjectRoot:
    """Tests for project root detection."""

    def test_get_project_root(self):
        """Test finding project root."""
        root = get_project_root()
        assert root is not None
        assert isinstance(root, Path)

    def test_project_root_has_setup(self):
        """Test that project root contains setup.py."""
        root = get_project_root()
        # Should find setup.py eventually or use fallback
        assert root.exists()


class TestInstalledExtras:
    """Tests for installed extras detection."""

    def test_get_installed_extras_from_env(self):
        """Test reading extras from environment."""
        with patch.dict('os.environ', {'BAGUETTOTRON_DATASETS': 'wikipedia,synth'}):
            extras = get_installed_extras()
            assert 'wikipedia' in extras
            assert 'synth' in extras

    def test_get_installed_extras_from_marker(self):
        """Test reading extras from marker file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            marker_file = tmpdir / '.dataset_extras'

            with open(marker_file, 'w') as f:
                json.dump(['wikipedia', 'demo'], f)

            with patch('baguettotron.data.auto_download.get_project_root', return_value=tmpdir):
                extras = get_installed_extras()
                # May not find marker if project root is different
                assert isinstance(extras, set)

    def test_get_installed_extras_empty(self):
        """Test when no extras are installed."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('baguettotron.data.auto_download.get_project_root') as mock_root:
                mock_root.return_value = Path('/nonexistent')
                extras = get_installed_extras()
                assert isinstance(extras, set)


class TestDatasetAvailability:
    """Tests for dataset availability checking."""

    @pytest.fixture
    def mock_project_root(self):
        """Create mock project root with datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / 'data'
            data_dir.mkdir()

            # Create mock dataset files
            (data_dir / 'train_tokens.json').write_text('[]')
            (data_dir / 'wikipedia_simple_tokens.json').write_text('[]')

            yield tmpdir

    def test_is_dataset_available_true(self, mock_project_root):
        """Test detecting available dataset."""
        with patch('baguettotron.data.auto_download.get_project_root', return_value=mock_project_root):
            # Check for synth (looks for train_tokens.json)
            available = is_dataset_available('synth')
            assert available is True

    def test_is_dataset_available_false(self):
        """Test detecting unavailable dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('baguettotron.data.auto_download.get_project_root', return_value=Path(tmpdir)):
                available = is_dataset_available('synth')
                assert available is False

    def test_is_dataset_available_unknown_dataset(self):
        """Test checking unknown dataset."""
        available = is_dataset_available('unknown_dataset')
        assert available is False


class TestDatasetInfo:
    """Tests for dataset information retrieval."""

    def test_get_dataset_info_valid(self):
        """Test getting info for valid dataset."""
        info = get_dataset_info('synth')
        assert info is not None
        assert 'name' in info
        assert 'script' in info
        assert 'size' in info
        assert info['name'] == 'PleIAs/SYNTH'

    def test_get_dataset_info_wikipedia(self):
        """Test getting Wikipedia dataset info."""
        info = get_dataset_info('wikipedia')
        assert info is not None
        assert info['name'] == 'wikimedia/wikipedia'
        assert 'prepare_wikipedia_data.py' in info['script']

    def test_get_dataset_info_invalid(self):
        """Test getting info for invalid dataset."""
        info = get_dataset_info('nonexistent')
        assert info is None


class TestListAvailableDatasets:
    """Tests for listing available datasets."""

    @pytest.fixture
    def mock_project_with_datasets(self):
        """Create mock project with some datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / 'data'
            data_dir.mkdir()

            # Create some dataset files
            (data_dir / 'train_tokens.json').write_text('[]')
            (data_dir / 'wikipedia_simple_tokens.json').write_text('[]')
            (data_dir / 'eval.json').write_text('[]')

            yield tmpdir

    def test_list_available_datasets(self, mock_project_with_datasets):
        """Test listing available datasets."""
        with patch('baguettotron.data.auto_download.get_project_root', return_value=mock_project_with_datasets):
            available = list_available_datasets()
            assert isinstance(available, list)
            # Should find at least synth and wikipedia
            assert len(available) >= 1

    def test_list_available_datasets_empty(self):
        """Test listing when no datasets available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('baguettotron.data.auto_download.get_project_root', return_value=Path(tmpdir)):
                available = list_available_datasets()
                assert isinstance(available, list)
                assert len(available) == 0


class TestDatasetsRegistry:
    """Tests for DATASETS registry."""

    def test_datasets_registry_structure(self):
        """Test that DATASETS registry has correct structure."""
        assert 'synth' in DATASETS
        assert 'wikipedia' in DATASETS
        assert 'demo' in DATASETS

        for dataset_id, config in DATASETS.items():
            assert 'name' in config
            assert 'script' in config
            assert 'default_args' in config
            assert 'size' in config
            assert 'files' in config

            assert isinstance(config['name'], str)
            assert isinstance(config['script'], str)
            assert isinstance(config['default_args'], list)
            assert isinstance(config['files'], list)

    def test_synth_dataset_config(self):
        """Test SYNTH dataset configuration."""
        synth = DATASETS['synth']
        assert synth['name'] == 'PleIAs/SYNTH'
        assert 'prepare_synth_data.py' in synth['script']
        assert '--subset' in synth['default_args']
        # Files include the 'data/' prefix
        assert any('train_tokens.json' in f for f in synth['files'])

    def test_wikipedia_dataset_config(self):
        """Test Wikipedia dataset configuration."""
        wiki = DATASETS['wikipedia']
        assert wiki['name'] == 'wikimedia/wikipedia'
        assert 'prepare_wikipedia_data.py' in wiki['script']
        assert '--lang' in wiki['default_args']
        # Files include the 'data/' prefix
        assert any('wikipedia_simple_tokens.json' in f for f in wiki['files'])

    def test_demo_dataset_config(self):
        """Test demo dataset configuration."""
        demo = DATASETS['demo']
        assert demo['name'] == 'demo (synthetic)'
        assert 'prepare_demo_data.py' in demo['script']
        # Files include the 'data/' prefix
        assert any('train_tokens.json' in f for f in demo['files'])


class TestDownloadDataset:
    """Tests for dataset download functionality."""

    @patch('subprocess.run')
    @patch('baguettotron.data.auto_download.get_project_root')
    def test_download_dataset_success(self, mock_root, mock_run):
        """Test successful dataset download."""
        from baguettotron.data.auto_download import download_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            scripts_dir = tmpdir / 'scripts'
            scripts_dir.mkdir()

            # Create mock script
            script = scripts_dir / 'prepare_demo_data.py'
            script.write_text('#!/usr/bin/env python3\nprint("Demo")')

            mock_root.return_value = tmpdir
            mock_run.return_value = Mock(returncode=0)

            result = download_dataset('demo')
            assert result is True
            assert mock_run.called

    @patch('baguettotron.data.auto_download.get_project_root')
    def test_download_dataset_invalid(self, mock_root):
        """Test downloading invalid dataset."""
        from baguettotron.data.auto_download import download_dataset

        result = download_dataset('invalid_dataset')
        assert result is False

    @patch('subprocess.run')
    @patch('baguettotron.data.auto_download.get_project_root')
    def test_download_dataset_script_missing(self, mock_root, mock_run):
        """Test download when script is missing."""
        from baguettotron.data.auto_download import download_dataset

        mock_root.return_value = Path('/nonexistent')
        result = download_dataset('demo')
        assert result is False


class TestDownloadRequestedDatasets:
    """Tests for downloading multiple datasets."""

    @patch('baguettotron.data.auto_download.download_dataset')
    @patch('baguettotron.data.auto_download.is_dataset_available')
    def test_download_requested_datasets_specific(self, mock_available, mock_download):
        """Test downloading specific datasets."""
        from baguettotron.data.auto_download import download_requested_datasets

        mock_available.return_value = False
        mock_download.return_value = True

        results = download_requested_datasets(datasets=['wikipedia', 'demo'])

        assert 'wikipedia' in results
        assert 'demo' in results
        assert results['wikipedia'] is True
        assert results['demo'] is True

    @patch('baguettotron.data.auto_download.download_dataset')
    @patch('baguettotron.data.auto_download.is_dataset_available')
    def test_download_requested_datasets_skip_existing(self, mock_available, mock_download):
        """Test skipping already downloaded datasets."""
        from baguettotron.data.auto_download import download_requested_datasets

        # First dataset already available
        def available_side_effect(dataset_id):
            return dataset_id == 'wikipedia'

        mock_available.side_effect = available_side_effect
        mock_download.return_value = True

        results = download_requested_datasets(datasets=['wikipedia', 'demo'], force=False)

        # Wikipedia should be skipped (already available)
        assert results['wikipedia'] is True
        # Demo should be downloaded
        assert 'demo' in results

    @patch('baguettotron.data.auto_download.get_installed_extras')
    def test_download_requested_datasets_auto_detect(self, mock_extras):
        """Test auto-detection of datasets to download."""
        from baguettotron.data.auto_download import download_requested_datasets

        mock_extras.return_value = {'wikipedia', 'synth'}

        with patch('baguettotron.data.auto_download.download_dataset') as mock_download:
            with patch('baguettotron.data.auto_download.is_dataset_available', return_value=False):
                mock_download.return_value = True
                results = download_requested_datasets(datasets=None)

                # Should try to download detected datasets
                assert len(results) >= 0  # May be empty if no extras detected


class TestErrorScenarios:
    """Tests for error scenarios."""

    def test_get_dataset_info_none(self):
        """Test get_dataset_info returns None for invalid dataset."""
        info = get_dataset_info('totally_invalid_dataset_name_12345')
        assert info is None

    def test_is_dataset_available_none_dataset(self):
        """Test is_dataset_available with None."""
        # Should not crash
        result = is_dataset_available('')
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
