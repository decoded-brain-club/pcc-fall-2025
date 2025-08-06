import sys
from pathlib import Path
import os
import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src")) # Add src directory to path

from dataset_loader import DatasetLoader

class TUHLoader(DatasetLoader):
    def __init__(self, config, raw: bool = True):
        super().__init__(config, raw, config.cap)

    @staticmethod
    def _extract_subject_id(filename: str) -> str:
        basename = os.path.basename(filename)  # Get just the filename without path
        return basename.split('_')[0]
    
    def _generate_split_report(self, train_files: List[str], val_files: List[str], test_files: List[str]):
        """Generate a detailed report of the data splits"""
        
        def get_subject_stats(file_list):
            subjects = [self._extract_subject_id(f) for f in file_list]
            unique_subjects = list(set(subjects))
            return {
                'n_files': len(file_list),
                'n_subjects': len(unique_subjects),
                'subjects': sorted(unique_subjects),
                'files_per_subject': {s: subjects.count(s) for s in unique_subjects}
            }
        
        train_stats = get_subject_stats(train_files)
        val_stats = get_subject_stats(val_files)
        test_stats = get_subject_stats(test_files)
        
        # Save split report
        log_path = Path(__file__).parent.parent.parent / self.config.log_path
        report_path = os.path.join(log_path, "data_split_report.txt")
        os.makedirs(log_path, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("DATA SPLIT REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"TRAINING SET:\n")
            f.write(f"  Files: {train_stats['n_files']}\n")
            f.write(f"  Subjects: {train_stats['n_subjects']}\n")
            f.write(f"  Subject IDs: {', '.join(train_stats['subjects'][:10])}{'...' if len(train_stats['subjects']) > 10 else ''}\n\n")
            
            f.write(f"VALIDATION SET:\n")
            f.write(f"  Files: {val_stats['n_files']}\n")
            f.write(f"  Subjects: {val_stats['n_subjects']}\n")
            f.write(f"  Subject IDs: {', '.join(val_stats['subjects'][:10])}{'...' if len(val_stats['subjects']) > 10 else ''}\n\n")
            
            f.write(f"TEST SET:\n")
            f.write(f"  Files: {test_stats['n_files']}\n")
            f.write(f"  Subjects: {test_stats['n_subjects']}\n")
            f.write(f"  Subject IDs: {', '.join(test_stats['subjects'][:10])}{'...' if len(test_stats['subjects']) > 10 else ''}\n\n")
            
            # verify no overlap
            train_subjects = set(train_stats['subjects'])
            val_subjects = set(val_stats['subjects'])
            test_subjects = set(test_stats['subjects'])
            
            f.write("SUBJECT ISOLATION VERIFICATION:\n")
            f.write(f"  Train ∩ Val: {len(train_subjects & val_subjects)} subjects\n")
            f.write(f"  Train ∩ Test: {len(train_subjects & test_subjects)} subjects\n")
            f.write(f"  Val ∩ Test: {len(val_subjects & test_subjects)} subjects\n")
            
            if len(train_subjects & val_subjects) == 0 and len(train_subjects & test_subjects) == 0 and len(val_subjects & test_subjects) == 0:
                f.write("✓ NO DATA LEAKAGE\n")
            else:
                f.write("⚠ DATA LEAKAGE DETECTED\n")
        
        print(f"Detailed split report saved to: {report_path}")

    def _split(self) -> Tuple[List[str], List[str], List[str]]:
        """Split the dataset into training, validation, and testing sets."""

        seed = self.config.file_shuffle_seed # For reproducibility

        subject_to_files = defaultdict(list)
        for file in self.file_paths:
            subject_id = self._extract_subject_id(file)
            subject_to_files[subject_id].append(file)
        
        unique_subjects = sorted(list(subject_to_files.keys()))
        n_subjects = len(unique_subjects)

        files_per_subject = [len(files) for files in subject_to_files.values()]
        print(f"Files per subject - Mean: {np.mean(files_per_subject):.1f}, "
            f"Std: {np.std(files_per_subject):.1f}, "
            f"Min: {min(files_per_subject)}, Max: {max(files_per_subject)}")
        
        random.seed(seed)
        np.random.seed(seed)

        shuffled_subjects = unique_subjects.copy()
        random.shuffle(shuffled_subjects)
        
        # Split indices
        split_config = self.config.data_split_config
        train_end = int(n_subjects * split_config['training'])
        val_end = train_end + int(n_subjects * split_config['validation'])
        
        # Subject assignments
        train_subjects = shuffled_subjects[:train_end]
        val_subjects = shuffled_subjects[train_end:val_end]
        test_subjects = shuffled_subjects[val_end:]
        
        train_files = []
        val_files = []
        test_files = []
        
        for subject in train_subjects:
            train_files.extend(subject_to_files[subject])
        
        for subject in val_subjects:
            val_files.extend(subject_to_files[subject])
            
        for subject in test_subjects:
            test_files.extend(subject_to_files[subject])
        
        # sort for deterministic ordering
        train_files.sort()
        val_files.sort()
        test_files.sort()
        
        # verification
        all_train_subjects = set(self._extract_subject_id(f) for f in train_files)
        all_val_subjects = set(self._extract_subject_id(f) for f in val_files)
        all_test_subjects = set(self._extract_subject_id(f) for f in test_files)
        
        assert len(all_train_subjects & all_val_subjects) == 0, "Subject leakage between train and val!"
        assert len(all_train_subjects & all_test_subjects) == 0, "Subject leakage between train and test!"  
        assert len(all_val_subjects & all_test_subjects) == 0, "Subject leakage between val and test!"

        # Generate split report
        self._generate_split_report(train_files, val_files, test_files)

        return train_files, val_files, test_files