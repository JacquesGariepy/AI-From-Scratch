#!/usr/bin/env python3
"""
Tests for multi-dataset functionality.

Tests:
- MultiDataset loading and combining
- Weighted sampling
- Auto-detection
- Dataset configurations
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from baguettotron.data.multi_dataset import (
    MultiDataset,
    DatasetConfig,
    auto_detect_datasets,
    create_multi_dataset,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_dataset_config_creation(self):
        """Test creating a dataset configuration."""
        config = DatasetConfig(
            name="test_dataset",
            path=Path("data/test.json"),
            weight=0.7,
            max_samples=1000
        )

        assert config.name == "test_dataset"
        assert config.path == Path("data/test.json")
        assert config.weight == 0.7
        assert config.max_samples == 1000

    def test_dataset_config_defaults(self):
        """Test default values."""
        config = DatasetConfig(
            name="test",
            path=Path("data/test.json")
        )

        assert config.weight == 1.0
        assert config.max_samples is None


class TestMultiDataset:
    """Tests for MultiDataset class."""

    @pytest.fixture
    def temp_datasets(self):
        """Create temporary dataset files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create dataset 1 (100 samples)
            data1 = [[1, 2, 3, 4, 5] for _ in range(100)]
            dataset1_path = tmpdir / "dataset1.json"
            with open(dataset1_path, 'w') as f:
                json.dump(data1, f)

            # Create dataset 2 (50 samples)
            data2 = [[6, 7, 8, 9, 10] for _ in range(50)]
            dataset2_path = tmpdir / "dataset2.json"
            with open(dataset2_path, 'w') as f:
                json.dump(data2, f)

            # Create JSONL dataset (30 samples)
            dataset3_path = tmpdir / "dataset3.jsonl"
            with open(dataset3_path, 'w') as f:
                for i in range(30):
                    json.dump({'text': f'Sample {i}', 'tokens': [11, 12, 13]}, f)
                    f.write('\n')

            yield tmpdir, dataset1_path, dataset2_path, dataset3_path

    def test_multi_dataset_creation(self, temp_datasets):
        """Test creating a multi-dataset."""
        tmpdir, ds1, ds2, _ = temp_datasets

        configs = [
            DatasetConfig("ds1", ds1, weight=0.7),
            DatasetConfig("ds2", ds2, weight=0.3),
        ]

        dataset = MultiDataset(configs, block_size=10)

        assert len(dataset.datasets) == 2
        assert len(dataset) == 150  # 100 + 50
        assert len(dataset.dataset_weights) == 2
        assert abs(dataset.dataset_weights[0] - 0.7) < 0.01
        assert abs(dataset.dataset_weights[1] - 0.3) < 0.01

    def test_multi_dataset_getitem(self, temp_datasets):
        """Test getting items from multi-dataset."""
        tmpdir, ds1, ds2, _ = temp_datasets

        configs = [
            DatasetConfig("ds1", ds1, weight=0.5),
            DatasetConfig("ds2", ds2, weight=0.5),
        ]

        dataset = MultiDataset(configs, block_size=10)

        # Get item from first dataset
        item = dataset[0]
        assert 'input_ids' in item
        assert 'labels' in item
        assert 'dataset_name' in item
        assert isinstance(item['input_ids'], torch.Tensor)
        assert item['dataset_name'] in ['ds1', 'ds2']

    def test_multi_dataset_weighted_sampler(self, temp_datasets):
        """Test weighted sampler creation."""
        tmpdir, ds1, ds2, _ = temp_datasets

        configs = [
            DatasetConfig("ds1", ds1, weight=0.7),
            DatasetConfig("ds2", ds2, weight=0.3),
        ]

        dataset = MultiDataset(configs, block_size=10)
        sampler = dataset.get_weighted_sampler()

        assert sampler is not None
        assert len(list(sampler)) == len(dataset)

    def test_multi_dataset_max_samples(self, temp_datasets):
        """Test max_samples limiting."""
        tmpdir, ds1, _, _ = temp_datasets

        configs = [
            DatasetConfig("ds1", ds1, weight=1.0, max_samples=10),
        ]

        dataset = MultiDataset(configs, block_size=10)

        assert len(dataset.datasets[0]) == 10  # Limited to 10

    def test_multi_dataset_jsonl_support(self, temp_datasets):
        """Test JSONL format support."""
        tmpdir, _, _, ds3 = temp_datasets

        configs = [
            DatasetConfig("ds3", ds3, weight=1.0),
        ]

        dataset = MultiDataset(configs, block_size=10)

        assert len(dataset.datasets[0]) == 30
        item = dataset[0]
        assert 'input_ids' in item


class TestAutoDetection:
    """Tests for auto-detection functionality."""

    @pytest.fixture
    def mock_data_dir(self):
        """Create mock data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock dataset files
            (tmpdir / "train_tokens.json").write_text("[]")
            (tmpdir / "wikipedia_simple_tokens.json").write_text("[]")
            (tmpdir / "wikipedia_fr_tokens.json").write_text("[]")

            yield tmpdir

    def test_auto_detect_datasets(self, mock_data_dir):
        """Test auto-detecting datasets."""
        configs = auto_detect_datasets(mock_data_dir)

        assert len(configs) >= 2  # Should find at least synth and wikipedia_simple
        dataset_names = [c.name for c in configs]
        assert 'synth' in dataset_names or 'wikipedia_simple' in dataset_names

    def test_auto_detect_empty_dir(self):
        """Test auto-detection in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = auto_detect_datasets(tmpdir)
            assert len(configs) == 0


class TestCreateMultiDataset:
    """Tests for create_multi_dataset factory function."""

    @pytest.fixture
    def mock_data_dir(self):
        """Create mock data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock datasets
            data = [[1, 2, 3] for _ in range(10)]

            (tmpdir / "synth_tokens.json").write_text(json.dumps(data))
            (tmpdir / "wikipedia_tokens.json").write_text(json.dumps(data))

            yield tmpdir

    def test_create_multi_dataset_auto(self, mock_data_dir):
        """Test creating with auto-detection."""
        dataset = create_multi_dataset(
            datasets='auto',
            data_dir=mock_data_dir,
            block_size=10
        )

        assert len(dataset.datasets) >= 1

    def test_create_multi_dataset_specific(self, mock_data_dir):
        """Test creating with specific datasets."""
        dataset = create_multi_dataset(
            datasets=['synth', 'wikipedia'],
            data_dir=mock_data_dir,
            block_size=10
        )

        assert len(dataset.datasets) == 2

    def test_create_multi_dataset_with_weights(self, mock_data_dir):
        """Test creating with custom weights."""
        dataset = create_multi_dataset(
            datasets=['synth', 'wikipedia'],
            weights=[0.7, 0.3],
            data_dir=mock_data_dir,
            block_size=10
        )

        assert abs(dataset.dataset_weights[0] - 0.7) < 0.01
        assert abs(dataset.dataset_weights[1] - 0.3) < 0.01

    def test_create_multi_dataset_comma_separated(self, mock_data_dir):
        """Test creating with comma-separated string."""
        dataset = create_multi_dataset(
            datasets='synth,wikipedia',
            data_dir=mock_data_dir,
            block_size=10
        )

        assert len(dataset.datasets) == 2


class TestMultiDatasetIntegration:
    """Integration tests for multi-dataset training."""

    @pytest.fixture
    def temp_datasets(self):
        """Create temporary datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple datasets
            for i in range(3):
                data = [[j for j in range(5)] for _ in range(20)]
                path = tmpdir / f"dataset{i}.json"
                with open(path, 'w') as f:
                    json.dump(data, f)

            yield tmpdir

    def test_dataloader_with_multi_dataset(self, temp_datasets):
        """Test creating DataLoader with multi-dataset."""
        configs = [
            DatasetConfig("ds0", temp_datasets / "dataset0.json", weight=0.5),
            DatasetConfig("ds1", temp_datasets / "dataset1.json", weight=0.3),
            DatasetConfig("ds2", temp_datasets / "dataset2.json", weight=0.2),
        ]

        dataset = MultiDataset(configs, block_size=10)
        sampler = dataset.get_weighted_sampler()

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler
        )

        # Test iteration
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 4  # batch size

    def test_multi_dataset_statistics(self, temp_datasets):
        """Test dataset statistics and mixing."""
        configs = [
            DatasetConfig("ds0", temp_datasets / "dataset0.json", weight=0.6),
            DatasetConfig("ds1", temp_datasets / "dataset1.json", weight=0.4),
        ]

        dataset = MultiDataset(configs, block_size=10)

        # Check statistics
        assert len(dataset) == 40  # 20 + 20
        assert len(dataset.datasets) == 2
        assert dataset.dataset_names == ["ds0", "ds1"]
        assert len(dataset.sample_weights) == 40

        # Weights should be normalized
        total_weight = sum(dataset.dataset_weights)
        assert abs(total_weight - 1.0) < 0.01


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_dataset_file(self):
        """Test handling of missing dataset files."""
        configs = [
            DatasetConfig("missing", Path("/nonexistent/file.json"), weight=1.0),
        ]

        with pytest.raises(ValueError, match="No datasets could be loaded"):
            MultiDataset(configs, block_size=10)

    def test_invalid_format(self):
        """Test handling of invalid file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            invalid_file = tmpdir / "invalid.txt"
            invalid_file.write_text("not json")

            configs = [
                DatasetConfig("invalid", invalid_file, weight=1.0),
            ]

            with pytest.raises(ValueError):
                MultiDataset(configs, block_size=10)

    def test_create_multi_dataset_no_datasets(self):
        """Test error when no datasets found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No datasets found"):
                create_multi_dataset('auto', data_dir=tmpdir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
