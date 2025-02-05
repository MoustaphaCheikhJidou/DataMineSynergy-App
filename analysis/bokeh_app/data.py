# bokeh_app/data.py
import pandas as pd
from django.apps import apps
from django.db.models import Avg, Q

# Access Django models (make sure your app is in INSTALLED_APPS)
DrillHole = apps.get_model('analysis', 'DrillHole')
DrillInterval = apps.get_model('analysis', 'DrillInterval')
ChemicalAnalysis = apps.get_model('analysis', 'ChemicalAnalysis')
ElementValue = apps.get_model('analysis', 'ElementValue')

def get_uranium_by_depth():
    """
    Calculates the average uranium content for each 1-meter depth interval.
    """
    # Get all intervals with associated chemical analyses
    intervals = DrillInterval.objects.prefetch_related('chemical_analyses__element_values')

    data = []
    for interval in intervals:
        for analysis in interval.chemical_analyses.all():
            u_value = analysis.element_values.filter(element='U', unit='ppm').first()
            if u_value:
                avg_depth = (interval.depth_from + interval.depth_to) / 2
                data.append({
                    'depth': avg_depth,
                    'U_ppm': u_value.value,
                    'lithology': interval.lithology,
                    'hole_id': interval.drill_hole_id  
                })

    df = pd.DataFrame(data)
    return df

def get_element_correlations_for_granite(element='U'):
    """
    Calculates correlations between a given element and other elements in granite samples.
    """
    granite_intervals = DrillInterval.objects.filter(lithology='granite').prefetch_related(
        'chemical_analyses__element_values'
    )

    element_data = []
    for interval in granite_intervals:
        analysis = interval.chemical_analyses.first()  # Assuming one analysis per interval
        if analysis:
            values = {ev.element: ev.value for ev in analysis.element_values.all()}
            element_data.append(values)

    df = pd.DataFrame(element_data)
    # Replace non-numeric values with NaN before correlation calculation
    df = df.apply(pd.to_numeric, errors='coerce')
    
    if element not in df.columns:
        return pd.DataFrame()

    correlations = df.corr()[element].reset_index()
    correlations.columns = ['Element', 'Correlation']
    return correlations

def get_lithology_uranium_distribution():
    """
    Gets the distribution of uranium concentrations for each lithology.
    """
    intervals = DrillInterval.objects.prefetch_related('chemical_analyses__element_values')

    data = []
    for interval in intervals:
        for analysis in interval.chemical_analyses.all():
            u_value = analysis.element_values.filter(element='U', unit='ppm').first()
            if u_value:
                data.append({
                    'U_ppm': u_value.value,
                    'lithology': interval.lithology,
                    'hole_id': interval.drill_hole_id
                })

    df = pd.DataFrame(data)
    return df

def get_major_elements_vs_si02():
    """
    Gets major element concentrations against SiO2 for granite.
    """
    granite_intervals = DrillInterval.objects.filter(
        lithology='granite', depth_from__gte=6
    ).prefetch_related('chemical_analyses__element_values')

    data = []
    for interval in granite_intervals:
        analysis = interval.chemical_analyses.first()
        if analysis:
            element_values = {ev.element: ev.value for ev in analysis.element_values.all()}
            if 'SiO2' in element_values:
                data.append(element_values)

    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def get_trace_elements_vs_si02():
    """
    Gets trace element concentrations against SiO2 for granite.
    """
    granite_intervals = DrillInterval.objects.filter(
        lithology='granite', depth_from__gte=6
    ).prefetch_related('chemical_analyses__element_values')

    data = []
    for interval in granite_intervals:
        analysis = interval.chemical_analyses.first()
        if analysis:
            element_values = {ev.element: ev.value for ev in analysis.element_values.all()}
            if 'SiO2' in element_values:
                data.append(element_values)

    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def get_scatter_data_for_element(element1, element2):
    """
    Gets data for a scatter plot of two elements.
    """
    granite_intervals = DrillInterval.objects.filter(
        lithology='granite', depth_from__gte=6
    ).prefetch_related('chemical_analyses__element_values')

    data = []
    for interval in granite_intervals:
        analysis = interval.chemical_analyses.first()
        if analysis:
            element_values = {ev.element: ev.value for ev in analysis.element_values.all()}
            if element1 in element_values and element2 in element_values:
                data.append({
                    element1: element_values[element1],
                    element2: element_values[element2]
                })

    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def get_stats_uranium_concentration():
    """
    Calculates basic statistics on uranium concentrations.
    """
    intervals = DrillInterval.objects.prefetch_related('chemical_analyses__element_values')

    data = []
    for interval in intervals:
        for analysis in interval.chemical_analyses.all():
            u_value = analysis.element_values.filter(element='U', unit='ppm').first()
            if u_value:
                data.append({
                    'U_ppm': u_value.value,
                    'lithology': interval.lithology,
                    'depth_from': interval.depth_from,
                    'depth_to': interval.depth_to
                })

    df = pd.DataFrame(data)
    if df.empty:
        return None

    # Calculate statistics
    stats = df.groupby('lithology')['U_ppm'].agg(['mean', 'std', 'min', 'max'])
    stats = stats.rename(columns={
        'mean': 'Average U (ppm)',
        'std': 'Standard Deviation',
        'min': 'Min U (ppm)',
        'max': 'Max U (ppm)'
    })

    return stats