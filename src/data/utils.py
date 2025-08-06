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
            'AF3', 'AF4', 'AF7', 'AFF5h', 'AFF6h', 'AFz',
            'F1', 'F2', 'F3', 'F4', 'F5', 'F7', 'Fz',
            'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
            'FCC3h', 'FCC4h', 'FCC5h', 'FCC6h', 'FCCz',
            'FFC3h', 'FFC4h', 'FFC5h', 'FFC6h',
            'Fp1', 'Fp2', 'Fpz'
        ],
        'temporal': [
            'FFT7h', 'FFT8h', 'FTT7h', 'FTT8h',
            'Ft7', 'T7', 'T8',
            'TP7', 'TP8',
            'TPP5h', 'TTP7h', 'TTP8h'
        ],
        'central': [
            'C1', 'C2', 'C3', 'C4', 'Cz',
            'CCP1h', 'CCP2h', 'CCP3h', 'CCP4h'
        ],
        'parietal': [
            'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
            'CP3', 'CP4',
            'CPP5h', 'CPP6h',
            'POz'
        ]
    }


# Legacy electrode mappings (keep for backward compatibility)
frontal_electrodes = get_electrode_regions()['frontal']
temporal_electrodes = get_electrode_regions()['temporal'] 
central_electrodes = get_electrode_regions()['central']
parietal_electrodes = get_electrode_regions()['parietal']