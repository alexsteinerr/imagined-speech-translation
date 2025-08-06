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
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'FC5', 'FC1', 'FC2', 'FC6', 'AF3', 'AF4', 'F1', 'F2', 'F5', 'F6'
        ],
        'temporal': [
            'T7', 'T8', 'TP9', 'TP10', 'FT9', 'FT10', 
            'T9', 'T10', 'FT7', 'FT8', 'TP7', 'TP8'
        ],
        'central': [
            'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 
            'FC3', 'FC4', 'C1', 'C2', 'C5', 'C6'
        ],
        'parietal': [
            'P7', 'P3', 'Pz', 'P4', 'P8', 'CP3', 'CP4', 
            'PO9', 'O1', 'Oz', 'O2', 'PO10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'P1', 'P2', 'P5', 'P6', 'PO1', 'PO2'
        ]
    }

# Legacy electrode mappings (keep for backward compatibility)
frontal_electrodes = get_electrode_regions()['frontal']
temporal_electrodes = get_electrode_regions()['temporal'] 
central_electrodes = get_electrode_regions()['central']
parietal_electrodes = get_electrode_regions()['parietal']