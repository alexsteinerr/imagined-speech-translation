"""
Utility functions for EEG data processing.
"""

def get_electrode_regions():
    """
    Get electrode mappings for different brain regions.
    
    Returns:
        Dict mapping region names to lists of electrode names
    """
    return {
        'frontal': [
            'FC5', 'F5', 'F7', 'F3', 'FC1', 'F1', 'AF3', 'Fz',
            'FC2', 'F2', 'AF4', 'Fp2', 'F4', 'F6', 'F8', 'FC6'
        ],
        'temporal': [
            'T9', 'FT9', 'T7', 'TP7', 'FT8', 'T10', 'FT10', 'T8', 'TP8'
        ],
        'central': [
            'C5', 'C3', 'FC3', 'C1', 'CP1', 'Cz',
            'CP2', 'C2', 'C4', 'FC4', 'C6'
        ],
        'parietal': [
            'P7', 'P5', 'CP3', 'P3', 'PO3', 'PO1',
            'PO2', 'P4', 'PO4', 'P6', 'CP4', 'P8'
        ]
    }


# Legacy electrode mappings (keep for backward compatibility)
frontal_electrodes = get_electrode_regions()['frontal']
temporal_electrodes = get_electrode_regions()['temporal'] 
central_electrodes = get_electrode_regions()['central']
parietal_electrodes = get_electrode_regions()['parietal']