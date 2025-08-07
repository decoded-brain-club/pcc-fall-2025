import os
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import logging

class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths to absolute paths"""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            return str(self.project_root / path_obj)
    
    @property
    def raw_data_path(self) -> str:
        return self._resolve_path(self.config['data']['raw_data_path'])
    
    @property
    def raw_training_epochs_path(self) -> str:
        return self._resolve_path(self.config['data']['raw_training_epochs_path'])
    
    @property
    def raw_validation_epochs_path(self) -> str:
        return self._resolve_path(self.config['data']['raw_validation_epochs_path'])
    
    @property
    def raw_testing_epochs_path(self) -> str:
        return self._resolve_path(self.config['data']['raw_testing_epochs_path'])
    
    @property
    def clean_training_epochs_path(self) -> str:
        return self._resolve_path(self.config['data']['clean_training_epochs_path'])
    
    @property
    def clean_validation_epochs_path(self) -> str:
        return self._resolve_path(self.config['data']['clean_validation_epochs_path'])
    
    @property
    def clean_testing_epochs_path(self) -> str:
        return self._resolve_path(self.config['data']['clean_testing_epochs_path'])
    
    @property
    def matlab_workspace_path(self) -> str:
        return self._resolve_path(self.config['data']['matlab_workspace_path'])

    @property
    def caploc_file_path(self) -> str:
        return self._resolve_path(self.config['data']['caploc_file_path'])

    @property
    def cap(self) -> Optional[int]:
        cap_value = self.config['data'].get('cap')
        if cap_value is not None:
            return int(cap_value)
        return None
    
    @property
    def eeglab_absolute_path(self) -> str:
        return self._resolve_path(self.config['plugin']['eeglab_absolute_path'])

    @property
    def log_path(self) -> str:
        return self._resolve_path(self.config['logging']['log_path'])
    
    @property
    def epoch_duration(self) -> int:
        return self.config['processing']['epoch_duration']
    
    @property
    def step_duration(self) -> int:
        return self.config['processing']['step_duration']
    
    @property
    def target_sfreq(self) -> int:
        return self.config['processing']['target_sfreq']
    
    @property
    def minimum_time_per_weight(self) -> int:
        return self.config['processing']['minimum_time_per_weight']
    
    @property
    def file_shuffle_seed(self) -> int:
        return self.config['processing']['file_shuffle_seed']
    
    @property
    def random_state_seed(self) -> int:
        return self.config['processing']['random_state_seed']
    
    @property
    def mwf_passes(self) -> int:
        return self.config['processing']['mwf_passes']

    @property
    def expected_channels(self) -> int:
        return self.config['processing']['expected_channels']

    @property
    def exclude_channels(self) -> List[str]:
        """List of channel names to exclude from processing."""
        return self.config['processing'].get('exclude_channels', [])

    @property
    def filter_config(self) -> Dict[str, float]:
        return self.config['filters']
    
    @property
    def data_split_config(self) -> Dict[str, float]:
        return self.config['data_split']

# global config instance
_config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config