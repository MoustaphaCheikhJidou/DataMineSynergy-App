import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
from django.core.exceptions import ValidationError
from django.apps import apps
import tempfile
import re
import logging
import io
import os
import tempfile
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from .models import *
from django.shortcuts import render
from django.conf import settings

from bokeh.plotting import figure
from bokeh.embed import components, file_html
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    CategoricalColorMapper,
    NumeralTickFormatter,
    DataTable,
    TableColumn,
    TabPanel,
    Tabs,
    LinearColorMapper,
    ColorBar,
    BasicTicker,
    PrintfTickFormatter,
    Select,
    Button,
    Div,
    WMTSTileSource,
    NumberFormatter
)
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Category10, Plasma256, Inferno256, Magma256
from bokeh.resources import CDN

from scipy.stats import gaussian_kde
from pyproj import Transformer

from django.db.models import Avg, Q
from django.contrib import messages
from django.http import HttpResponse
from django.template.loader import render_to_string
# from xhtml2pdf import pisa   ## REMOVE THIS
from datetime import datetime

# Your local modules
from .forms import UploadFileForm
from .models import DrillInterval, ChemicalAnalysis, ElementValue


from reportlab.lib import colors
from reportlab.lib.pagesizes import letter,landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import numpy as np
from pyproj import Transformer
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, NumeralTickFormatter
from bokeh.palettes import Plasma256, Viridis256, Inferno256
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, WMTSTileSource, HoverTool
from bokeh.layouts import column
from bokeh.models import TabPanel
import numpy.ma as ma

from django.contrib.auth.decorators import login_required
# Configure le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_field_name(sheet_name, column_name):
    """Map Excel column names to Django model field names."""
    # Retrieve the mapping for the given sheet
    # sheet_mapping = COLUMN_MAPPING.get(sheet_name, {})
    # Return the mapped field name if exists, else apply default sanitization
    return default_sanitize(column_name)

def default_sanitize(name):
    """Default sanitization: convert to snake_case."""
    name = re.sub(r'[^\w\s]', '', name)  # Remove special characters except underscores
    name = re.sub(r'[\s\-]+', '_', name)  # Replace spaces and hyphens with underscores
    name = name.lower()
    # Ensure it doesn't start with a number
    if re.match(r'^\d', name):
        name = f'_{name}'
    return name

def sanitize_sheet_name(name):
    """Convert sheet names to valid Python class names."""
    name = re.sub(r'\W|^(?=\d)', '_', name)
    return ''.join(word.capitalize() for word in name.split('_'))

def select_lithology(row):
    """
    Fonction pour choisir la lithologie avec le plus grand pourcentage
    """
    if pd.isna(row['Lith1_pct']) and pd.isna(row['Lith2_pct']):
        return row['Lithology1']
    elif pd.isna(row['Lith2_pct']) or (not pd.isna(row['Lith1_pct']) and row['Lith1_pct'] >= row['Lith2_pct']):
        return row['Lithology1']
    else:
        return row['Lithology2']

def process_merged_data(merged_df):
    """
    Traite les données fusionnées et les enregistre dans les nouvelles tables relationnelles
    """
    logger.info("Colonnes disponibles dans le DataFrame:")
    logger.info(merged_df.columns.tolist())

    # Standardiser les noms de colonnes
    column_mapping = {
        'HOLEID': 'holeid',
        'HoleID': 'holeid',
        'Hole_ID': 'holeid',
        'PROJECT': 'project',
        'Project': 'project',
        'PROSPECT': 'prospect',
        'Prospect': 'prospect',
        'EASTING': 'easting',
        'Easting': 'easting',
        'NORTHING': 'northing',
        'Northing': 'northing',
        'DEPTH_FROM': 'depth_from',
        'DepthFrom': 'depth_from',
        'Depth_From': 'depth_from',
        'DEPTH_TO': 'depth_to',
        'DepthTo': 'depth_to',
        'Depth_To': 'depth_to',
        'LITHOLOGY': 'lithology',
        'Lithology': 'lithology',
        'Lithology1': 'lithology'
    }

    # Renommer les colonnes
    merged_df = merged_df.rename(columns=column_mapping)

    logger.info("Colonnes après standardisation:")
    logger.info(merged_df.columns.tolist())

    with transaction.atomic():
        # Création des trous de forage uniques
        for hole_id in merged_df['holeid'].unique():
            hole_data = merged_df[merged_df['holeid'] == hole_id].iloc[0]
            
            # Convertir hole_id en string s'il est numérique
            hole_id_str = str(hole_id)
            
            drill_hole = DrillHole.objects.create(
                hole_id=hole_id_str,
                project=hole_data.get('project'),
                prospect=hole_data.get('prospect'),
                easting=hole_data.get('easting'),
                northing=hole_data.get('northing')
            )

        # Traitement des intervalles et analyses
        for _, row in merged_df.iterrows():
            # Convertir hole_id en string
            hole_id_str = str(row['holeid'])
            
            # Création de l'intervalle
            interval = DrillInterval.objects.create(
                drill_hole_id=hole_id_str,
                depth_from=row['depth_from'],
                depth_to=row['depth_to'],
                lithology=row.get('lithology')
            )

            # Préparer les données d'analyse chimique
            chemical_data = {}
            
            # Mapper les noms de colonnes pour les éléments majeurs
            major_elements = {
                'SiO2_PCT': 'sio2',
                'Al2O3_PCT': 'al2o3',
                'Fe2O3_PCT': 'fe2o3',
                'CaO_PCT': 'cao',
                'MgO_PCT': 'mgo',
                'Na2O_PCT': 'na2o',
                'K2O_PCT': 'k2o',
                'TiO2_PCT': 'tio2',
                'P2O5_PCT': 'p2o5',
                'LOI_PCT': 'loi'
            }

            # Mapper les noms de colonnes pour les éléments traces
            trace_elements = {
                'Au_PPM': 'au',
                'Ag_PPM': 'ag',
                'Cu_PPM': 'cu',
                'Pb_PPM': 'pb',
                'Zn_PPM': 'zn',
                'As_PPM': 'as_val',
                'Ni_PPM': 'ni',
                'Co_PPM': 'co',
                'Cr_PPM': 'cr',
                'V_PPM': 'v'
            }

            # Ajouter les éléments majeurs
            for excel_name, db_name in major_elements.items():
                for possible_name in [excel_name, excel_name.lower(), excel_name.title()]:
                    if possible_name in row:
                        chemical_data[db_name] = row[possible_name]
                        break

            # Ajouter les éléments traces
            for excel_name, db_name in trace_elements.items():
                for possible_name in [excel_name, excel_name.lower(), excel_name.title()]:
                    if possible_name in row:
                        chemical_data[db_name] = row[possible_name]
                        break

            # Création de l'analyse chimique
            analysis = ChemicalAnalysis.objects.create(
                interval=interval,
                **chemical_data
            )

            # Traitement des autres éléments chimiques
            for col in row.index:
                # Vérifier si c'est une colonne d'analyse chimique
                if any(col.upper().endswith(suffix) for suffix in ['_PPM', '_PCT', '_PCB']):
                    # Ignorer les colonnes déjà traitées
                    if col.upper() not in major_elements and col.upper() not in trace_elements:
                        value = row[col]
                        if pd.notna(value):
                            # Extraire le nom de l'élément et l'unité
                            parts = col.split('_')
                            element = parts[0].upper()
                            unit = parts[-1].lower()
                            
                            ElementValue.objects.create(
                                analysis=analysis,
                                element=element,
                                value=value,
                                unit=unit
                            )

def process_aura_file(file_path):
    try:
        logger.info("Début du traitement du fichier Excel")

        # Charger les feuilles dans des DataFrames
        logger.info("Lecture des feuilles Excel...")
        try:
            df_collars = pd.read_excel(file_path, sheet_name="Collars")  # Assuming 'Collars' sheet exists
            df_geology = pd.read_excel(file_path, sheet_name="DHGeology")
            df_assays = pd.read_excel(file_path, sheet_name="DHAssays")
            logger.info(f"Feuilles chargées avec succès. Collars: {len(df_collars)} lignes, Géologie: {len(df_geology)} lignes, Analyses: {len(df_assays)} lignes")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des feuilles Excel: {str(e)}")
            raise ValueError(f"Erreur lors de la lecture des feuilles Excel. Assurez-vous que les feuilles 'Collars', 'DHGeology' et 'DHAssays' existent. Erreur: {str(e)}")

        # Standardiser les noms de colonnes pour df_collars
        collars_column_mapping = {
            'HOLEID': 'HoleID',
            'Hole_ID': 'HoleID',
            'holeid': 'HoleID',
            'HOLE_ID': 'HoleID',
            'PROJECT': 'Project',
            'project': 'Project',
            'PROSPECT': 'Prospect',
            'prospect': 'Prospect',
            'EASTING': 'Easting',
            'easting': 'Easting',
            'NORTHING': 'Northing',
            'northing': 'Northing'
        }
        df_collars = df_collars.rename(columns=collars_column_mapping)

        # Standardiser les noms de colonnes (existing code)
        column_mapping = {
            # ... (your existing column mapping)
        }
        df_geology = df_geology.rename(columns=column_mapping)
        df_assays = df_assays.rename(columns=column_mapping)

        # Vérifier les colonnes requises
        required_collars_columns = ['HoleID', 'Project', 'Easting', 'Northing']  # Add 'Easting', 'Northing'
        required_geology_columns = ['HoleID', 'Project', 'DepthFrom', 'DepthTo', 'Lithology1']
        required_assays_columns = ['HoleID', 'Project', 'DepthFrom', 'DepthTo']
        missing_collars = [col for col in required_collars_columns if col not in df_collars.columns]
        missing_geology = [col for col in required_geology_columns if col not in df_geology.columns]
        missing_assays = [col for col in required_assays_columns if col not in df_assays.columns]
        if missing_collars or missing_geology or missing_assays:
            raise ValueError(f"Colonnes manquantes - Collars: {missing_collars}, Géologie: {missing_geology}, Analyses: {missing_assays}")

        # Select only necessary columns and ensure they exist
        df_collars = df_collars[required_collars_columns]
        df_geology = df_geology[['Project', 'Prospect', 'HoleID', 'DepthFrom', 'DepthTo', 'Lithology1']]

        # Merge Collars with Geology
        df_merged_collars_geology = pd.merge(df_geology, df_collars, on=['HoleID', 'Project'], how='left')

        # Comparer les colonnes 'HoleID' dans les deux DataFrames
        comparison = df_merged_collars_geology['HoleID'].isin(df_assays['HoleID'])
        df_merged_collars_geology = df_merged_collars_geology[comparison]

        # Comparer les HoleID dans df_assays et df_geology
        comparison_assays = df_assays['HoleID'].isin(df_merged_collars_geology['HoleID'])
        df_assays = df_assays[comparison_assays]

        # Suppression des NaN uniquement dans la colonne 'U_ppm'
        if 'U_ppm' in df_assays.columns:
            df_assays = df_assays[df_assays['U_ppm'].notna()]

        # Créer une colonne ID en combinant HoleID, DepthFrom et DepthTo
        df_merged_collars_geology['ID'] = df_merged_collars_geology['HoleID'].astype(str) + df_merged_collars_geology['DepthFrom'].astype(str) + df_merged_collars_geology['DepthTo'].astype(str)
        df_assays['ID'] = df_assays['HoleID'].astype(str) + df_assays['DepthFrom'].astype(str) + df_assays['DepthTo'].astype(str)

        # Sélectionner les colonnes d'analyses chimiques
        assay_cols = ['ID', 'Project', 'HoleID', 'DepthFrom', 'DepthTo'] + [col for col in df_assays.columns if any(element in col for element in ['_pct', '_ppm', '_ppb'])]
        df_assays = df_assays[assay_cols]

        # Réinitialiser les indices
        df_merged_collars_geology = df_merged_collars_geology.reset_index(drop=True)
        df_assays = df_assays.reset_index(drop=True)

        # Trouver les ID communs
        common_ids = df_merged_collars_geology['ID'].isin(df_assays['ID'])
        df_merged_collars_geology_common = df_merged_collars_geology[common_ids]

        common_ids_assays = df_assays['ID'].isin(df_merged_collars_geology_common['ID'])
        df_assays_common = df_assays[common_ids_assays]

        # Identifier et supprimer les colonnes communes de df_geology_common
        common_columns = set(df_merged_collars_geology_common.columns).intersection(set(df_assays_common.columns)) - {'ID'}
        df_merged_collars_geology_common = df_merged_collars_geology_common.drop(columns=common_columns)

        # Effectuer la fusion
        df_merged = pd.merge(df_assays_common, df_merged_collars_geology_common, on='ID', how='left')

        # Convertir toutes les valeurs de Lithology1 en minuscules
        df_merged['Lithology1'] = df_merged['Lithology1'].str.lower()

        # Afficher les informations sur les données fusionnées
        logger.info(f"Colonnes dans df_merged: {df_merged.columns.tolist()}")
        logger.info(f"Nombre de lignes dans df_merged: {len(df_merged)}")

        # Sauvegarder les données dans la base de données
        with transaction.atomic():
            # Supprimer les anciennes données from the tables
            DrillHole.objects.all().delete()
            DrillInterval.objects.all().delete()
            ChemicalAnalysis.objects.all().delete()
            ElementValue.objects.all().delete()
            logger.info("Anciennes données supprimées")
            
            # Create DrillHoles
            unique_holes = df_merged[['HoleID', 'Project', 'Prospect', 'Easting', 'Northing']].drop_duplicates()
            logger.info(f"Nombre de trous uniques: {len(unique_holes)}")
            
            for index, row in unique_holes.iterrows():
                try: 
                   DrillHole.objects.create(
                        hole_id=str(row.HoleID),
                        project=str(row.Project),
                        prospect=str(row.Prospect) if pd.notna(row.Prospect) else None,
                        easting=float(row.Easting) if pd.notna(row.Easting) else None,  # Handle missing values
                        northing=float(row.Northing) if pd.notna(row.Northing) else None   # Handle missing values
                    )
                except Exception as e:
                    logger.error(f"Erreur lors de la création du trou {row.HoleID}: {str(e)}")
                    raise
            # Create DrillIntervals
            for index, row in df_merged.iterrows():
                try:
                    drill_hole = DrillHole.objects.get(hole_id=str(row.HoleID))
                    
                    interval = DrillInterval.objects.create(
                        drill_hole=drill_hole,
                        depth_from=float(row.DepthFrom),
                        depth_to=float(row.DepthTo),
                        lithology=str(row.Lithology1)
                    )
                    # Create ChemicalAnalysis
                    chemical_analysis = ChemicalAnalysis.objects.create(
                        interval=interval
                    )

                    # Add ElementValue for Chemical Analysis
                    for col in df_merged.columns:
                        if any(suffix in col.lower() for suffix in ['_ppm', '_ppb', '_pct']):
                            value = row[col]
                            if pd.notna(value):
                                try:
                                    element = col.split('_')[0]
                                    unit = col.split('_')[1]
                                    # Ensure the value is non-negative
                                    value = max(0.0, float(value))
                                    ElementValue.objects.create(
                                        analysis=chemical_analysis,
                                        element=element,
                                        value=float(value),
                                        unit=unit
                                    )
                                except Exception as e:
                                    logger.error(f"Erreur lors de la création de ElementValue pour {col}: {str(e)}")
                                    continue
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de l'intervalle pour le trou {row.HoleID}: {str(e)}")
                    raise

        return True, "Les données ont été importées avec succès"

    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
        logger.exception("Détails de l'erreur:")
        return False, str(e)
    
@login_required
def upload_excel(request):
    if request.method == 'POST' and request.FILES.get('file'):
        excel_file = request.FILES['file']
        logger.info(f"Fichier reçu : {excel_file.name}, taille : {excel_file.size} bytes")
        
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            for chunk in excel_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
            logger.info(f"Fichier temporaire créé : {temp_file_path}")

        try:
            success, message = process_aura_file(temp_file_path)
            if success:
                messages.success(request, f"Succès : {message}")
                logger.info("Traitement réussi")
            else:
                messages.error(request, f"Erreur : {message}")
                logger.error(f"Erreur lors du traitement : {message}")
        except Exception as e:
            error_message = f"Erreur lors du traitement du fichier : {str(e)}"
            messages.error(request, error_message)
            logger.error(error_message)
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_file_path)
                logger.info("Fichier temporaire supprimé")
            except Exception as e:
                logger.error(f"Erreur lors de la suppression du fichier temporaire : {str(e)}")
        
        return redirect('upload_excel')  # Redirect even if there's an error

    return render(request, 'upload.html')



def generate_report_pdf(request):
    # 1. Fetch Data (same data as used for the dashboard)
    intervals = DrillInterval.objects.select_related('drill_hole').prefetch_related('chemical_analyses__element_values').all()
    data_rows = []
    for interval in intervals:
        hole = interval.drill_hole
        avg_depth = (interval.depth_from + interval.depth_to) / 2.0
        for analysis in interval.chemical_analyses.all():
            element_map = {ev.element.upper(): ev.value for ev in analysis.element_values.all()}
            row_dict = {
                'hole_id': hole.hole_id,
                'depth': avg_depth,
                'lithology': interval.lithology.lower() if interval.lithology else "unknown",
                'Longitude': hole.easting,
                'Latitude': hole.northing,
                'U': element_map.get('U', 0),
                'TH': element_map.get('TH', 0),
                'V': element_map.get('V', 0),
                'SIO2': element_map.get('SIO2', 0),
                'FEO': element_map.get('FEO', 0) or element_map.get('FE2O3', 0),
                'AL2O3': element_map.get('AL2O3', 0),
                'CAO': element_map.get('CAO', 0),
                'MGO': element_map.get('MGO', 0),
                'K2O': element_map.get('K2O', 0),
                'NA2O': element_map.get('NA2O', 0),
                'TIO2': element_map.get('TIO2', 0),
                'BA': element_map.get('BA', 0),
                'NB': element_map.get('NB', 0),
                'RB': element_map.get('RB', 0),
                'SR': element_map.get('SR', 0),
                'ZN': element_map.get('ZN', 0),
                'ZR': element_map.get('ZR', 0),
                'LA': element_map.get('LA', 0),
                'AS': element_map.get('AS', 0),
                'CO': element_map.get('CO', 0),
                'CR': element_map.get('CR', 0),
                'CU': element_map.get('CU', 0),
                'GA': element_map.get('GA', 0),
                'GE': element_map.get('GE', 0),
                'LI': element_map.get('LI', 0),
                'MO': element_map.get('MO', 0),
                'NI': element_map.get('NI', 0),
                'PB': element_map.get('PB', 0),
                'SC': element_map.get('SC', 0),
                'TA': element_map.get('TA', 0),
                'W': element_map.get('W', 0),
                'Y': element_map.get('Y', 0),
            }
            data_rows.append(row_dict)

    df = pd.DataFrame(data_rows)

    # Ensure that all numeric columns are actually numeric:
    numeric_columns = df.select_dtypes(include=np.number).columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df_granite = df[df['lithology'] == 'granite']
    available_elements_scatter = ['U', 'TH', 'V', 'CAO', 'SIO2', 'FEO', 'AL2O3', 'K2O', 'NA2O', 'TIO2', 'BA', 'NB', 'RB', 'SR', 'ZN', 'ZR', 'LA', 'AS', 'CO', 'CR', 'CU', 'GA', 'GE', 'LI', 'MO', 'NI', 'PB', 'SC', 'TA', 'W', 'Y', 'MGO']

    # 2. Create Bokeh Plots (using your existing functions)
    uranium_depth_plot = plot_uranium_by_depth(df)
    correlation_plot = plot_element_correlations(df_granite, element='U')
    lithology_distribution_plot = plot_lithology_uranium_distribution(df)
    scatter_plot_panel = create_scatter_plot_panel(df_granite, available_elements_scatter)
    stats_table = display_uranium_statistics_table(df)
    geochemical_table = plot_geochemical_data_table(df)
    uranium_distribution_plot = plot_uranium_distribution(df)
    map_panel = create_geochemical_map(df)

    # 3. Render Plots to HTML using file_html
    plot_htmls = [
        file_html(uranium_depth_plot, CDN, "Uranium by Depth"),
        file_html(correlation_plot, CDN, "Element Correlations"),
        file_html(lithology_distribution_plot, CDN, "Uranium Distribution by Lithology"),
        file_html(scatter_plot_panel, CDN, "Scatter Plot Panel"),
        file_html(stats_table, CDN, "Uranium Statistics Table"),
        file_html(geochemical_table, CDN, "Geochemical Data Table"),
        file_html(uranium_distribution_plot, CDN, "Uranium Distribution Plot"),
        file_html(map_panel, CDN, "Geochemical Map Panel") if isinstance(map_panel, TabPanel) else file_html(Div(text="Map not available."), CDN, "Geochemical Map"),
    ]

     # 2. Create PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="geochemical_report.pdf"'
    buffer = io.BytesIO()
    
    # Use landscape orientation for better table display
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Add title and date
    title = Paragraph("Geochemical Analysis Report", styles['Heading1'])
    date_str = datetime.now().strftime("%Y-%m-%d")
    date = Paragraph(f"Report Generated: {date_str}", styles['Normal'])
    story.extend([title, date, Spacer(1, 0.3*inch)])
    
    # Add summary statistics instead of plots
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Create summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_data = [['Element', 'Mean', 'Median', 'Min', 'Max', 'Std Dev']]
    
    for col in numeric_cols:
        row = [
            col,
            f"{df[col].mean():.2f}",
            f"{df[col].median():.2f}",
            f"{df[col].min():.2f}",
            f"{df[col].max():.2f}",
            f"{df[col].std():.2f}"
        ]
        stats_data.append(row)
    
    # Create statistics table
    stats_table = Table(stats_data, colWidths=[1.2*inch]*6)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add lithology distribution
    story.append(Paragraph("Lithology Distribution", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    litho_dist = df['lithology'].value_counts()
    litho_data = [['Lithology', 'Count']]
    litho_data.extend([[k, str(v)] for k, v in litho_dist.items()])
    
    litho_table = Table(litho_data, colWidths=[2*inch, 1*inch])
    litho_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(litho_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add summary of uranium concentrations by lithology
    story.append(Paragraph("Uranium Concentration by Lithology", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    u_stats = df.groupby('lithology')['U'].agg(['mean', 'min', 'max']).round(2)
    u_data = [['Lithology', 'Mean U (ppm)', 'Min U (ppm)', 'Max U (ppm)']]
    u_data.extend([[index, str(row['mean']), str(row['min']), str(row['max'])] 
                   for index, row in u_stats.iterrows()])
    
    u_table = Table(u_data, colWidths=[1.5*inch]*4)
    u_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(u_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add footer
    story.append(Paragraph("End of Report", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    response.write(pdf_value)
    
    return response

def plot_uranium_by_depth(df):
    """
    Simple scatter + line for average U by integer depth interval.
    """
    if df.empty:
        return Div(text="No data available for Uranium vs Depth plot.")
    
    source = ColumnDataSource(df)

    p = figure(width=800, height=400,
               title="Average Uranium Concentration by Depth",
               x_axis_label="Average Depth (m)",
               y_axis_label="Uranium (ppm)")

    p.scatter(x='depth', y='U', size=8, source=source, alpha=0.5, color="blue")

    # Calculate average uranium concentration for each integer depth
    depth_min, depth_max = df['depth'].min(), df['depth'].max()
    depth_intervals = np.arange(int(np.floor(depth_min)), int(np.ceil(depth_max)) + 1)
    avg_u_by_depth = []
    for i in range(len(depth_intervals) - 1):
        low = depth_intervals[i]
        high = depth_intervals[i+1]
        interval_data = df[(df['depth'] >= low) & (df['depth'] < high)]
        avg_u = interval_data['U'].mean() if not interval_data.empty else None
        avg_u_by_depth.append(avg_u)

    avg_source = ColumnDataSource(data={
        'depth': depth_intervals[:-1],
        'avg_U': avg_u_by_depth
    })

    # Add a line for the average uranium concentration
    p.line(x='depth', y='avg_U', line_width=2, color="red", source=avg_source,
           legend_label="Avg U by 1m interval")

    hover = HoverTool(tooltips=[
        ("Depth", "@depth m"),
        ("Uranium", "@U ppm"),
        ("Lithology", "@lithology"),
        ("Hole ID", "@hole_id")
    ])
    p.add_tools(hover)

    p.legend.location = "top_right"

    return p

def plot_element_correlations(df, element='U'):
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # If the selected element doesn't exist in numeric_df, bail out
    if element not in numeric_df.columns:
        return Div(text=f"No numeric data available for {element} correlations.")

    corr_matrix = numeric_df.corr()
    corr_series = corr_matrix.get(element)

    if corr_series is None:
        return Div(text=f"No correlation found for {element} in numeric columns.")

    # Convert the correlation series to a DataFrame for plotting
    correlations = corr_series.reset_index()
    correlations.columns = ['Element', 'Correlation']
    source = ColumnDataSource(correlations)

    p = figure(width=800, height=500,
               title=f"Correlation of Elements with {element}",
               x_range=list(correlations['Element']),
               y_axis_label="Correlation Coefficient",
               toolbar_location="above")

    mapper = linear_cmap('Correlation', palette=Plasma256, low=-1, high=1)

    p.vbar(x='Element', top='Correlation', width=0.9,
           source=source, line_color="white", fill_color=mapper)

    hover = HoverTool(tooltips=[
        ("Element", "@Element"),
        ("Correlation", "@Correlation{0.000}")
    ])
    p.add_tools(hover)

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 1.2

    return p

def plot_lithology_uranium_distribution(df):
    """
    Create multiple kernel density lines for each lithology's U distribution.
    """
    if df.empty:
        return Div(text="No data available for Uranium Distribution by Lithology plot.")

    unique_lithologies = df['lithology'].unique().tolist()
    # If too many lithologies, palette might not be large enough, handle that
    palette = Category10[max(3, min(len(unique_lithologies), 10))]

    p = figure(width=800, height=400,
               x_axis_label="Uranium (ppm)",
               y_axis_label="Density",
               title="Uranium Distribution by Lithology")

    # For each lithology, compute KDE if enough data
    for i, lith in enumerate(unique_lithologies):
        lith_df = df[df['lithology'] == lith]
        if len(lith_df) < 2:
            continue
        density = gaussian_kde(lith_df['U'])
        xvals = np.linspace(0, lith_df['U'].max(), 300)
        yvals = density(xvals)
        # Plot line
        p.line(xvals, yvals, line_width=2, color=palette[i % len(palette)],
               legend_label=lith)

    hover = HoverTool(tooltips=[
        ("Uranium", "$x ppm"),
        ("Density", "$y"),
    ], mode='vline')
    p.add_tools(hover)

    p.legend.location = "top_right"

    return p

def plot_lithology_counts(df):
    """
    Creates a bar plot showing the count of each lithology.

    Args:
        df: The DataFrame containing the data.

    Returns:
        A Bokeh plot object.
    """
    if df.empty or 'lithology' not in df.columns:
        return Div(text="<p>No data available for Lithology Counts plot.</p>")

    # Count the occurrences of each lithology
    lithology_counts = df['lithology'].value_counts()

    source = ColumnDataSource(data={
        'lithology': lithology_counts.index.tolist(),
        'counts': lithology_counts.values
    })

    p = figure(x_range=lithology_counts.index.tolist(), width=800, height=400,
               title="Count of Samples by Lithology",
               tools="pan,box_zoom,reset,hover,save")

    p.vbar(x='lithology', top='counts', width=0.9, source=source,
           line_color="white", fill_color=factor_cmap('lithology', palette=Category20[len(lithology_counts)], factors=lithology_counts.index.tolist()))

    p.xaxis.axis_label = "Lithology"
    p.yaxis.axis_label = "Count"
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    hover = HoverTool(tooltips=[("Lithology", "@lithology"), ("Count", "@counts")])
    p.add_tools(hover)

    return p

def plot_histogram_selected_elements(df, elements):
    """
    Creates histograms for a selection of elements.

    Args:
        df: DataFrame containing the geochemical data.
        elements: List of element column names to plot.

    Returns:
        A Bokeh layout object containing the histograms.
    """
    plots = []
    for elem in elements:
        if elem in df.columns:
            df[elem] = pd.to_numeric(df[elem], errors='coerce')
            df_filtered = df.dropna(subset=[elem])
            
            if not df_filtered.empty:
                hist, edges = np.histogram(df_filtered[elem], density=True, bins=50)

                p = figure(title=f"Distribution of {elem}", x_axis_label=elem, y_axis_label="Density",
                           tools="pan,box_zoom,reset,save")
                p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                       fill_color="skyblue", line_color="white", alpha=0.8)

                hover = HoverTool(tooltips=[(elem, "@"+elem+"{0.00}")])
                p.add_tools(hover)

                plots.append(p)
            else:
                print(f"Skipping {elem} due to no valid data.")
        else:
            print(f"Column {elem} not found in DataFrame.")

    return column(*plots)

def create_scatter_plot_panel(df, available_elements):
    """
    A dynamic scatter plot tab with fixed JavaScript callback.
    """
    if df.empty:
        return TabPanel(child=Div(text="No data for Scatter Plot."), title="Scatter Plot")
    
    from bokeh.models import Select, HoverTool, CustomJS, ColumnDataSource
    from bokeh.layouts import column, row
    from bokeh.plotting import figure

    # Clean out any elements not in df columns
    valid_els = [el for el in available_elements if el in df.columns]
    if not valid_els:
        return TabPanel(child=Div(text="No valid elements found in DataFrame."), title="Scatter Plot")

    # Create the full data source once
    source = ColumnDataSource(df)

    # Set initial x and y values
    initial_x = valid_els[0]
    initial_y = valid_els[1] if len(valid_els) > 1 else valid_els[0]

    # Create plot
    p = figure(width=800, height=500,
               title=f"Scatter: {initial_x} vs {initial_y}",
               tools="pan,box_zoom,reset,save",
               x_axis_label=initial_x,
               y_axis_label=initial_y)

    # Create scatter glyph
    scatter = p.scatter(x=initial_x, y=initial_y,
                       source=source, size=7, color="navy", alpha=0.5)

    # Add HoverTool separately
    hover = HoverTool(
        tooltips=[
            (initial_x, f"@{initial_x}"),
            (initial_y, f"@{initial_y}")
        ],
        renderers=[scatter]
    )
    p.add_tools(hover)

    # Create select widgets
    select_x = Select(title="X Axis", value=initial_x, options=valid_els)
    select_y = Select(title="Y Axis", value=initial_y, options=valid_els)

    # Create JavaScript callback with corrected axis handling
    callback = CustomJS(args=dict(p=p, scatter=scatter, source=source), code="""
        const x_name = cb_obj.title === "X Axis" ? cb_obj.value : scatter.glyph.x.field;
        const y_name = cb_obj.title === "Y Axis" ? cb_obj.value : scatter.glyph.y.field;
        
        // Update scatter glyph
        scatter.glyph.x = {field: x_name};
        scatter.glyph.y = {field: y_name};
        
        // Update axis labels - using the proper axis references
        p.below[0].axis_label = x_name;  // x-axis
        p.left[0].axis_label = y_name;   // y-axis
        
        // Update plot title
        p.title.text = "Scatter: " + x_name + " vs " + y_name;
        
        // Find and update hover tool
        const hover = p.toolbar.tools.find(tool => tool.type == "HoverTool");
        if (hover) {
            hover.tooltips = [
                [x_name, '@' + x_name],
                [y_name, '@' + y_name]
            ];
        }
        
        // Trigger a redraw
        source.change.emit();
    """)

    # Add callbacks
    select_x.js_on_change('value', callback)
    select_y.js_on_change('value', callback)

    # Create layout
    layout = column(row(select_x, select_y), p)
    
    return TabPanel(child=layout, title="Scatter Plot")


def display_uranium_statistics_table(df):
    if df.empty:
        return Div(text="No data available for Uranium Statistics table.")

    # Calculate statistics
    stats = df.groupby('lithology')['U'].agg(['mean', 'std', 'min', 'max']).reset_index()
    stats = stats.rename(columns={
        'mean': 'Average U (ppm)',
        'std': 'Standard Deviation',
        'min': 'Min U (ppm)',
        'max': 'Max U (ppm)'
    })

    # Create a ColumnDataSource from the statistics DataFrame
    source = ColumnDataSource(stats)

    # Define the columns for the DataTable, using NumberFormatter
    columns = [
        TableColumn(field="lithology", title="Lithology"),
        TableColumn(field="Average U (ppm)", title="Average U (ppm)", formatter=NumberFormatter(format="0.00")),
        TableColumn(field="Standard Deviation", title="Standard Deviation", formatter=NumberFormatter(format="0.00")),
        TableColumn(field="Min U (ppm)", title="Min U (ppm)", formatter=NumberFormatter(format="0.00")),
        TableColumn(field="Max U (ppm)", title="Max U (ppm)", formatter=NumberFormatter(format="0.00"))
    ]

    # Create the DataTable
    data_table = DataTable(source=source, columns=columns, width=800, height=400)

    return data_table
def create_statistics_table_tab(df):
    table = display_uranium_statistics_table(df)
    return TabPanel(child=table, title="Uranium Statistics")

def plot_geochemical_data_table(df):
    """
    Full data table of all columns in df (capped at some width).
    """
    if df.empty:
        return Div(text="No data available for Geochemical Data Table.")

    # Limit to N columns if there are too many?
    all_cols = list(df.columns)
    if len(all_cols) > 40:
        # you can choose to limit or show partial
        all_cols = all_cols[:40]

    source = ColumnDataSource(df[all_cols])
    columns = [TableColumn(field=col, title=col) for col in all_cols]
    data_table = DataTable(source=source, columns=columns, width=800, height=600)
    return data_table

def create_geochemical_table_tab(df):
    geochemical_table = plot_geochemical_data_table(df)
    return TabPanel(child=geochemical_table, title="Geochemical Data")

def plot_uranium_distribution(df):
    """
    Histogram + density line for df['U'].
    """
    if df.empty or 'U' not in df.columns:
        return Div(text="No data available for Uranium Distribution.")

    hist, edges = np.histogram(df['U'], density=True, bins=50)
    p = figure(title="Uranium Distribution (Histogram + PDF)",
               tools="pan,box_zoom,reset,save",
               background_fill_color="#fafafa",
               width=800, height=400)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)

    # Density line
    kde = gaussian_kde(df['U'])
    x = np.linspace(df['U'].min(), df['U'].max(), 300)
    density = kde(x)
    p.line(x, density, line_color="#ff8888", line_width=3, alpha=0.7, legend_label="Density")

    p.y_range.start = 0
    p.xaxis.axis_label = 'U (ppm)'
    p.yaxis.axis_label = 'Density'
    p.legend.location = 'top_right'
    return p

def create_uranium_distribution_tab(df):
    dist_plot = plot_uranium_distribution(df)
    return TabPanel(child=dist_plot, title="U Distribution")


from bokeh.palettes import Viridis256  # Or any other suitable palette
from bokeh.models import LabelSet, Text
def plot_correlation_matrix(df, element, elements_list, plot_width=600, plot_height=600):
    """
    Generates a correlation matrix heatmap for a given element against a list of elements using Bokeh.

    Args:
        df: The DataFrame containing the geochemical data.
        element: The element to compare against (e.g., 'U' for Uranium).
        elements_list: A list of elements to include in the correlation matrix.
        plot_width: Width of the plot.
        plot_height: Height of the plot.

    Returns:
        A Bokeh plot object.
    """
    # Ensure the element is in the list for correlation calculation
    if element not in elements_list:
        elements_list.insert(0, element)

    # Select only the relevant columns and ensure they are numeric
    df_selected = df[elements_list]
    df_numeric = df_selected.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    df_numeric.dropna(inplace=True)

    # Check if DataFrame is empty after cleanup
    if df_numeric.empty:
        return Div(text="<p>No data available for correlation matrix after data cleanup.</p>")

    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr()

    # Prepare data for plotting
    elements = corr_matrix.columns.tolist()
    num_elements = len(elements)

    x = []
    y = []
    colors = []
    correlations = []

    for i in range(num_elements):
        for j in range(num_elements):
            x.append(elements[i])
            y.append(elements[j])
            correlation_value = corr_matrix.iloc[i, j]
            correlations.append(correlation_value)
            # Using the Viridis256 palette for color mapping
            color_value = int(255 * (correlation_value + 1) / 2)  # Scale to 0-255
            colors.append(Viridis256[min(max(color_value, 0), 255)])  # Ensure value is in the palette's range

    source = ColumnDataSource(data=dict(x=x, y=y, colors=colors, correlations=correlations))

    # Create the figure
    p = figure(
        title=f"Correlation Matrix: {element} vs. Other Elements",
        x_range=elements,
        y_range=list(reversed(elements)),
        x_axis_location="above",
        width=plot_width,
        height=plot_height,
        tools="hover,pan,box_zoom,reset,save",
        toolbar_location='below',
        tooltips=[('Element 1', '@x'), ('Element 2', '@y'), ('Correlation', '@correlations{0.00}')],
    )

    # Add the heatmap
    p.rect(
        x='x',
        y='y',
        width=1,
        height=1,
        source=source,
        fill_color='colors',
        line_color=None,
    )

    # Add a color bar
    color_mapper = LinearColorMapper(palette=Viridis256, low=-1, high=1)
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        orientation='horizontal',
        padding=5,  # Reduced padding
        major_label_text_font_size="10px"  # Smaller font size for labels
    )
    p.add_layout(color_bar, 'below')

    # Rotate x-axis labels for better readability
    p.xaxis.major_label_orientation = 3.14 / 4  # Rotate by 45 degrees

    # Customize the appearance
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10px"  # Smaller font size for axis labels
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_text_font_style = "bold"

    # Add correlation values as text on each cell
    text_source = ColumnDataSource(data=dict(x=x, y=y, correlations=[f"{c:.2f}" for c in correlations]))
    text_labels = LabelSet(x='x', y='y', text='correlations', text_align='center', text_baseline='middle',
                           text_font_size="8pt", source=text_source, text_color="black")
    p.add_layout(text_labels)

    return p

from bokeh.transform import dodge, factor_cmap
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
from bokeh.palettes import Category10, Category20
from bokeh.transform import factor_cmap
from bokeh.embed import components

def plot_uranium_by_lithology(df):
    """
    Creates a refined box plot and scatter plot of Uranium concentration by lithology.
    
    Args:
        df: DataFrame with columns 'lithology' and 'U' (Uranium concentration in ppm)
    Returns:
        Bokeh figure object
    """
    # Data preparation
    df['U'] = pd.to_numeric(df['U'], errors='coerce')
    df_filtered = df.dropna(subset=['U', 'lithology'])
    
    # Calculate statistics for box plots
    stats = df_filtered.groupby('lithology')['U'].agg(['mean', 'std', 'min', 'max'])
    stats['q1'] = df_filtered.groupby('lithology')['U'].quantile(0.25)
    stats['q2'] = df_filtered.groupby('lithology')['U'].quantile(0.5)
    stats['q3'] = df_filtered.groupby('lithology')['U'].quantile(0.75)
    
    # Refined color palette to match the image
    colors = {
        'alluvium': '#1f77b4',
        'calcrete': '#ff7f0e',
        'calsilicate': '#2ca02c',
        'colluvium': '#d62728',
        'granite': '#9467bd',
        'gravels': '#8c564b',
        'mafic rock': '#e377c2',
        'sand': '#7f7f7f',
        'sandstone': '#bcbd22',
        'saprolite': '#17becf',
        'sediment': '#ff9896',
        'syenite': '#c5b0d5'
    }
    
    # Create figure with refined settings
    p = figure(width=900, height=600,
              x_range=sorted(df_filtered['lithology'].unique()),
              y_range=(0, 1000),
              toolbar_location="above",
              tools="pan,box_zoom,wheel_zoom,reset,save")
    
    # Create ColumnDataSource for box plots with adjusted coordinates
    stats['x0'] = [x - 0.2 for x in range(len(stats))]
    stats['x1'] = [x + 0.2 for x in range(len(stats))]
    source_stats = ColumnDataSource(stats.reset_index())
    
    # Add thinner box plots
    p.vbar(x='lithology', top='q3', bottom='q1', width=0.3,
           source=source_stats,
           fill_color=factor_cmap('lithology', palette=list(colors.values()), 
                                factors=sorted(df_filtered['lithology'].unique())),
           line_color="black",
           alpha=0.5)
    
    # Add whiskers with refined styling
    p.segment(x0='lithology', x1='lithology', y0='q3', y1='max',
             source=source_stats, line_color="black", line_width=1)
    p.segment(x0='lithology', x1='lithology', y0='q1', y1='min',
             source=source_stats, line_color="black", line_width=1)
    
    # Add median lines with refined styling
    p.segment(x0='x0', x1='x1', y0='q2', y1='q2',
             source=source_stats, line_color="black", line_width=1.5)
    
    # Add scatter points with refined jittering
    for i, lithology in enumerate(sorted(df_filtered['lithology'].unique())):
        lithology_data = df_filtered[df_filtered['lithology'] == lithology]
        
        # Add jitter to x-coordinates to avoid overlapping points
        jitter = np.random.normal(0, 0.1, size=len(lithology_data))  # Small jitter
        x_coords = [i + j for j in jitter]  # Center around the lithology index
        
        p.circle(x=x_coords,
                y=lithology_data['U'],
                size=5,  # Adjust point size
                fill_color=colors.get(lithology, colors['granite']),
                fill_alpha=0.6,
                line_color=None)
    
    # Refined styling
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = '#e5e5e5'
    p.ygrid.grid_line_dash = [6, 4]
    p.ygrid.grid_line_alpha = 0.3
    
    # Axis styling
    p.xaxis.axis_label = "Lithology"
    p.yaxis.axis_label = "Uranium (ppm)"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "10pt"
    p.xaxis.major_label_orientation = 0.3
    
    # Remove toolbar logo and add minimal padding
    p.toolbar.logo = None
    p.min_border_left = 50
    p.min_border_right = 50
    
    # Add hover tool with refined tooltips
    hover = HoverTool(tooltips=[
        ("Lithology", "@lithology"),
        ("Uranium", "@y{0.0} ppm")
    ])
    p.add_tools(hover)
    
    return p

def create_geochemical_map(df):
    """
    Example WMTSTileSource map. Expects 'Longitude' and 'Latitude' columns in df.
    We'll convert from local projection to lat/lon to web mercator.
    """
    print("DataFrame Info Before Map Generation:")
    print(df.info())

    if 'Longitude' not in df.columns or 'Latitude' not in df.columns:
        print("Longitude or Latitude columns are missing from DataFrame.")
        return TabPanel(child=Div(text="No Longitude/Latitude columns for mapping."), title="Geochemical Map")

    print("Longitude Data:")
    print(df['Longitude'])
    print("Latitude Data:")
    print(df['Latitude'])

    print(f"Min Longitude: {df['Longitude'].min()}, Max Longitude: {df['Longitude'].max()}")
    print(f"Min Latitude: {df['Latitude'].min()}, Max Latitude: {df['Latitude'].max()}")

    # 1. Define your local CRS.
    local_crs = "EPSG:32629"   # *** USE YOUR CORRECT EPSG CODE HERE! (32628, 32629 or other) ***
    # 2. Create transformers
    transformer_local_to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    transformer_wgs84_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Convert local to lat/long and then to web mercator
    merc_x = []
    merc_y = []
    lat = []
    lon = []

    for easting, northing in zip(df['Longitude'], df['Latitude']):
        # convert easting and northing to latitude and longitude.
        l, la = transformer_local_to_wgs84.transform(easting, northing)
        # now convert that lat/long into web mercator
        x, y = transformer_wgs84_to_merc.transform(l, la)
        merc_x.append(x)
        merc_y.append(y)
        lat.append(la)
        lon.append(l)
    df['Mercator_X'] = merc_x
    df['Mercator_Y'] = merc_y
    df['Latitude_WGS'] = lat
    df['Longitude_WGS'] = lon


    # Calculate map bounds (for Mauritania) - adjust as needed
    min_lon = -17   # min longitude
    max_lon = -4    # max longitude
    min_lat = 15    # min latitude
    max_lat = 27    # max latitude
    x1,y1 = transformer_wgs84_to_merc.transform(min_lon, min_lat)
    x2,y2 = transformer_wgs84_to_merc.transform(max_lon, max_lat)

    p = figure(x_axis_type="mercator", y_axis_type="mercator",
               width=800, height=600, title="Geochemical Map (WebMercator)",
               x_range=(x1, x2), # Set the boundaries based on longitude
               y_range=(y1, y2)
               )

    tile_provider = WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/rastertiles/light_all/{Z}/{X}/{Y}.png",
    )
    p.add_tile(tile_provider)


   # Generate 2D histogram using numpy:
    x = np.array(df['Mercator_X'])
    y = np.array(df['Mercator_Y'])
    weights = np.array(df['U'])
    # Define the number of bins
    bins_num = 100
    # create a 2d histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins_num, weights=weights)

    from scipy.ndimage import gaussian_filter
    hist_smooth = gaussian_filter(hist, sigma=1)
    
    # Normalize the smoothed data
    hist_normalized = hist_smooth / np.max(hist_smooth) if np.max(hist_smooth) > 0 else hist_smooth

    # Mask the histogram where it's zero
    hist_masked = ma.masked_where(hist_normalized == 0, hist_normalized)

    # Generate a colormap for the heatmap
    color_mapper = LinearColorMapper(palette=Inferno256, low=0, high=1, nan_color='rgba(0,0,0,0)')

    # Plot the heatmap. we'll use the `image` plotting function
    p.image(
        image=[hist_masked],  # Use the masked histogram data
        x=xedges[0],
        y=yedges[0],
        dw=xedges[-1] - xedges[0],
        dh=yedges[-1] - yedges[0],
        color_mapper = color_mapper
        )


    color_bar = ColorBar(color_mapper=color_mapper,
                            ticker=BasicTicker(desired_num_ticks=10),
                            formatter = NumeralTickFormatter(format="0.0"),
                            label_standoff=6, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # After creating the histogram, create a ColumnDataSource
    source = ColumnDataSource(data={
        'image': [hist_masked],
        'raw_values': [hist]
    })
    
    # Update the image glyph to use the source
    p.image(image='image', x=xedges[0], y=yedges[0], 
            dw=xedges[-1] - xedges[0], dh=yedges[-1] - yedges[0],
            color_mapper=color_mapper, source=source)
    
    # Update hover tool
    hover = HoverTool(tooltips=[
        ("Uranium Concentration", "@image{0.00}"),
        ("x", "$x"),
        ("y", "$y")
    ])
    p.add_tools(hover)


    return TabPanel(child=p, title="Geochemical Map")

########################################
# MAIN DASHBOARD VIEW
########################################
@login_required
def dashboard_view(request):
    # 1) Handle file upload if POST
    status_message = ""
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, file.name)
                with open(temp_file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                success, message = process_aura_file(temp_file_path)
                if success:
                    status_message = "File uploaded and processed successfully."
                else:
                    status_message = f"Error processing file: {message}"
            except Exception as e:
                status_message = f"An error occurred: {str(e)}"
                traceback.print_exc()
            finally:
                # Cleanup temp
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
        else:
            status_message = "Form is not valid. Please upload a valid XLSX file."
    else:
        form = UploadFileForm()

    # 2) Build DataFrame from DB
    intervals = DrillInterval.objects.select_related('drill_hole').prefetch_related('chemical_analyses__element_values').all()
    data_rows = []
    for interval in intervals:
        hole = interval.drill_hole
        avg_depth = (interval.depth_from + interval.depth_to) / 2.0
        for analysis in interval.chemical_analyses.all():
            element_map = {ev.element.upper(): ev.value for ev in analysis.element_values.all()}
            row_dict = {
                'hole_id': hole.hole_id,
                'depth': avg_depth,
                'lithology': interval.lithology.lower() if interval.lithology else "unknown",
                # Example placeholders for lat/lon if you have them
                'Longitude': hole.easting,  # Replace with your actual field if any
                'Latitude': hole.northing,   # Replace with your actual field if any
                'U': element_map.get('U', 0),
                'TH': element_map.get('TH', 0),
                'V': element_map.get('V', 0),
                'SIO2': element_map.get('SIO2', 0),
                'FEO': element_map.get('FEO', 0) or element_map.get('FE2O3', 0),
                'AL2O3': element_map.get('AL2O3', 0),
                'CAO': element_map.get('CAO', 0),
                'MGO': element_map.get('MGO', 0),
                'K2O': element_map.get('K2O', 0),
                'NA2O': element_map.get('NA2O', 0),
                'TIO2': element_map.get('TIO2', 0),
                'BA': element_map.get('BA', 0),
                'NB': element_map.get('NB', 0),
                'RB': element_map.get('RB', 0),
                'SR': element_map.get('SR', 0),
                'ZN': element_map.get('ZN', 0),
                'ZR': element_map.get('ZR', 0),
            }
            data_rows.append(row_dict)

    df = pd.DataFrame(data_rows)

    # 3) Create Bokeh Panels/Tabs
    # (a) Uranium vs Depth
    tab_uranium_depth = TabPanel(child=plot_uranium_by_depth(df), title="Uranium vs Depth")

    # (b) Correlations: we specifically look at the 'granite' subset, or entire df
    df_granite = df[df['lithology'] == 'granite']
    tab_correlation = TabPanel(child=plot_element_correlations(df_granite, element='U'), title="Element Correlations")

    # (c) Lithology distribution
    tab_lithology_distribution = TabPanel(child=plot_lithology_uranium_distribution(df), 
                                          title="Uranium Dist. by Lithology")
        # Before calling the function, check your data:
    # Check the actual ranges of both trace and major elements
    print("Trace Elements ranges:")
    for elem in ['BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']:
        print(f"{elem}: {df_granite[elem].min():.1f} to {df_granite[elem].max():.1f}")
    
    print("\nMajor Elements ranges:")
    for elem in ['SIO2', 'AL2O3', 'CAO', 'FEO', 'MGO', 'NA2O', 'K2O', 'TIO2']:
        print(f"{elem}: {df_granite[elem].min():.1f} to {df_granite[elem].max():.1f}")
    
    # Check for any non-zero values in major elements
    major_nonzero = df_granite[['SIO2', 'AL2O3', 'CAO', 'FEO', 'MGO', 'NA2O', 'K2O', 'TIO2']].any().any()
    print("\nAre there any non-zero values in major elements?", major_nonzero)
        # (d) Major elements vs. SiO2 (granite subset or entire df if you prefer)
    
    # (f) Scatter plot with dropdown
    available_elements = ['U', 'TH', 'V', 'SIO2', 'FEO', 'AL2O3', 'CAO', 'MGO',
                          'K2O', 'NA2O', 'TIO2', 'BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']
    scatter_plot_panel = create_scatter_plot_panel(df_granite, available_elements)

    # (g) Uranium stats table
    stats_table_tab = create_statistics_table_tab(df)

    # (h) Raw geochemical table
    geochemical_table_tab = create_geochemical_table_tab(df)

    # (i) Uranium distribution histogram + PDF
    uranium_dist_tab = create_uranium_distribution_tab(df)

    # (j) Map tab
    map_tab = create_geochemical_map(df)

    # Combine in a single Tabs
    tabs = Tabs(tabs=[
        tab_uranium_depth,
        tab_correlation,
        tab_lithology_distribution,
        scatter_plot_panel,
        stats_table_tab,
        geochemical_table_tab,
        uranium_dist_tab,
        map_tab
    ])

    script, div = components(tabs)

    context = {
        'script': script,
        'div': div,
        'form': form,
        'status_message': status_message,
    }
    return render(request, 'dashboard.html', context)

import pandas as pd
import numpy as np
import tempfile, os, traceback
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from bokeh.embed import components
from bokeh.models import Tabs, Div
from .forms import UploadFileForm
from .models import DrillInterval


@login_required
def bokeh_charts_page(request, page_id):
    # --- Handle file upload if POST (if applicable) ---
    status_message = ""
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, file.name)
                with open(temp_file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                # process_aura_file is assumed defined somewhere
                success, message = process_aura_file(temp_file_path)
                if success:
                    status_message = "File uploaded and processed successfully."
                else:
                    status_message = f"Error processing file: {message}"
            except Exception as e:
                status_message = f"An error occurred: {str(e)}"
                traceback.print_exc()
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
        else:
            status_message = "Form is not valid. Please upload a valid XLSX file."
    else:
        form = UploadFileForm()

    # --- Build DataFrame from the database ---
    intervals = DrillInterval.objects.select_related('drill_hole')\
                    .prefetch_related('chemical_analyses__element_values').all()
    data_rows = []
    for interval in intervals:
        hole = interval.drill_hole
        avg_depth = (interval.depth_from + interval.depth_to) / 2.0
        for analysis in interval.chemical_analyses.all():
            element_map = {ev.element.upper(): ev.value for ev in analysis.element_values.all()}
            row_dict = {
                'hole_id': hole.hole_id,
                'depth': avg_depth,
                'lithology': interval.lithology.lower() if interval.lithology else "unknown",
                'Longitude': hole.easting,  # adjust if necessary
                'Latitude': hole.northing,   # adjust if necessary
                'U': element_map.get('U', 0),
                'TH': element_map.get('TH', 0),
                'V': element_map.get('V', 0),
                'SIO2': element_map.get('SIO2', 0),
                'FEO': element_map.get('FEO', 0) or element_map.get('FE2O3', 0),
                'AL2O3': element_map.get('AL2O3', 0),
                'CAO': element_map.get('CAO', 0),
                'MGO': element_map.get('MGO', 0),
                'K2O': element_map.get('K2O', 0),
                'NA2O': element_map.get('NA2O', 0),
                'TIO2': element_map.get('TIO2', 0),
                'BA': element_map.get('BA', 0),
                'NB': element_map.get('NB', 0),
                'RB': element_map.get('RB', 0),
                'SR': element_map.get('SR', 0),
                'ZN': element_map.get('ZN', 0),
                'ZR': element_map.get('ZR', 0),
            }
            data_rows.append(row_dict)
    df = pd.DataFrame(data_rows)
    major_elements = {
        'AL2O3': ['AL2O3', 'Al2O3', 'al2o3'],
        'CAO': ['CAO', 'CaO', 'cao'],
        'FEO': ['FEO', 'FeO', 'feo'],
        'MGO': ['MGO', 'MgO', 'mgo'],
        'NA2O': ['NA2O', 'Na2O', 'na2o'],
        'K2O': ['K2O', 'k2o'],
        'TIO2': ['TIO2', 'TiO2', 'tio2']
    }
    trace_elements = ['BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']
     # --- Select charts based on page_id ---
    if page_id == "1":
        chart1 = plot_uranium_by_depth(df)
        chart2 = plot_element_correlations(df, element='U')
        page_title = "Uranium vs Depth and Element Correlations"

        script1, div1 = components(chart1)
        script2, div2 = components(chart2)
        context = {
            'page_title': page_title,
            'script1': script1,
            'div1': div1,
            'script2': script2,
            'div2': div2,
            'chart1_title' : chart1.title.text,
            'chart2_title' : chart2.title.text
        }
        
    elif page_id == "2":
        df_filtered = df.copy()
        df_filtered['U'] = pd.to_numeric(df_filtered['U'], errors='coerce')
        df_filtered.dropna(subset=['U'], inplace=True)
        chart1 = plot_lithology_uranium_distribution(df)
        chart2 = plot_uranium_by_lithology(df.copy())  # Use the new function
        page_title = "Lithology Distribution and Count"
       
        script1, div1 = components(chart1)
        script2, div2 = components(chart2)
        context = {
            'page_title': page_title,
             'script1': script1,
            'div1': div1,
            'script2': script2,
            'div2': div2,
            'chart1_title' : chart1.title.text,
            'chart2_title' : "Lithology Counts"
        }

    elif page_id == "3":
      trace_elements_list = ['BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']
      chart1 = plot_correlation_matrix(df.copy(), 'U', trace_elements_list)
      chart2 = plot_uranium_distribution(df)
      page_title = "Trace Elements vs SiO₂ and Uranium Distribution"

      script1, div1 = components(chart1)
      script2, div2 = components(chart2)
      context = {
          'page_title': page_title,
           'script1': script1,
          'div1': div1,
          'script2': script2,
          'div2': div2,
            'chart1_title' :  "Trace Element Distributions",
            'chart2_title' : chart2.title.text
      }
    elif page_id == "4":
        # For page 4, we want to use all element columns (i.e. all numeric columns except metadata)
        # Adjust the exclusion list as needed.
        exclude_columns = {'hole_id', 'depth', 'lithology', 'Longitude', 'Latitude'}
        available_elements = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_columns]
        # Now pass the complete available_elements list to the scatter plot panel
        panel1 = create_scatter_plot_panel(df, available_elements=available_elements)
        panel2 = create_statistics_table_tab(df)
        page_title = "Scatter Plot and Uranium Statistics"
        
        from bokeh.models import Tabs
        charts_tabs = Tabs(tabs=[panel1, panel2])
        script, div = components(charts_tabs)
        context = {
            'page_title': page_title,
            'script': script,
            'div': div,
        }
    
    elif page_id == "5":
        # Correlation matrix for Uranium vs. major elements
        trace_elements_list = ['BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']
        correlation_plot_major = plot_correlation_matrix(df.copy(), 'U', trace_elements_list)

        # Correlation matrix for Uranium vs. trace elements
        trace_elements_list = ['BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']
        correlation_plot_trace = plot_correlation_matrix(df.copy(), 'U', trace_elements_list)

        # Handle potential Div return for empty plots
        if isinstance(correlation_plot_major, Div):
            script_major, div_major = "", correlation_plot_major
        else:
            script_major, div_major = components(correlation_plot_major)

        if isinstance(correlation_plot_trace, Div):
            script_trace, div_trace = "", correlation_plot_trace
        else:
            script_trace, div_trace = components(correlation_plot_trace)

        # Update context
        context = {
            'script1': script_major,
            'div1': div_major,
            'chart1_title': "Correlation Matrix: Uranium vs. Major Elements",
            'script2': script_trace,
            'div2': div_trace,
            'chart4_title': "Correlation Matrix: Uranium vs. Trace Elements",
        }
    else:
         chart1 = plot_uranium_by_depth(df)
         chart2 = plot_element_correlations(df, element='U')
         page_title = "Default Charts"
    
         script1, div1 = components(chart1)
         script2, div2 = components(chart2)
         context = {
                'page_title': page_title,
                'script1': script1,
                'div1': div1,
                'script2': script2,
                'div2': div2,
                'chart1_title' : chart1.title.text,
                'chart2_title' : chart2.title.text
         }

    return render(request, 'bokeh_charts_page.html', context)



def loq(request):
    return render(request, 'base.html')
from django.shortcuts import render
from .models import DrillHole, DrillInterval, ElementValue
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    CategoricalColorMapper,
    WMTSTileSource,
    CustomJS
)
from bokeh.palettes import Category10, Category20, d3, Viridis256
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.embed import components
from bokeh.resources import CDN
from pyproj import Transformer
import plotly.graph_objects as go
from django.db.models import Max, Avg, Count
from django.http import JsonResponse
import json


def map_view(request):
    # 1. Data: fetch all holes
    drill_holes = DrillHole.objects.all()
    data = {
        'hole_id':   [h.hole_id  for h in drill_holes],
        'easting':   [h.easting  for h in drill_holes],
        'northing':  [h.northing for h in drill_holes],
    }
    df = pd.DataFrame(data)

    # 2. Convert from EPSG:32629 -> WebMercator
    transformer = Transformer.from_crs("EPSG:32629", "EPSG:3857", always_xy=True)
    merc_x, merc_y = transformer.transform(df['easting'].values, df['northing'].values)
    df['mercator_x'] = merc_x
    df['mercator_y'] = merc_y

    # 3. Bokeh data source
    source_2d = ColumnDataSource(df)

    # 4. Bokeh figure
    p = figure(
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_range=(df['mercator_x'].min(), df['mercator_x'].max()),
        y_range=(df['mercator_y'].min(), df['mercator_y'].max()),
        sizing_mode="stretch_width", 
        title="Drill Hole Locations",
        tools="pan,wheel_zoom,box_zoom,reset,tap"
    )
    tile_provider = WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/rastertiles/light_all/{Z}/{X}/{Y}.png"
    )
    p.add_tile(tile_provider)

    # 5. Plot hole circles
    p.circle(
        x='mercator_x',
        y='mercator_y',
        source=source_2d,
        size=10,
        color='blue',
        alpha=0.8
    )

    # 6. Hover
    hover = HoverTool(tooltips=[
        ("Hole ID", "@hole_id"),
        ("Easting", "@easting{0.00}"),
        ("Northing", "@northing{0.00}"),
    ])
    p.add_tools(hover)

    # 7. On tap, fetch two endpoints:
    #    /get_drill_hole_data/<hole_id>/ => 3D lines
    #    /get_3d_model_for_hole/<hole_id>/ => 3D volume
    callback = CustomJS(args=dict(source=source_2d), code='''
        const inds = cb_obj.indices;
        if (inds.length > 0) {
            const hole_id = source.data.hole_id[inds[0]];

            // 1) intervals & elements (vertical lines)
            fetch(`/get_drill_hole_data/${hole_id}/`)
              .then(r => r.json())
              .then(data => {
                  console.log("DrillHole intervals:", data);
                  window.create3DVisualization(data);
              });

            // 2) 3D IDW volume data
            fetch(`/get_3d_model_for_hole/${hole_id}/`)
              .then(r => r.json())
              .then(data => {
                  console.log("Volume data for hole:", data);
                  createVolumePlot(data);
              })
              .catch(err => console.error("Volume fetch error:", err));
        }
    ''')

    source_2d.selected.js_on_change('indices', callback)

    script, div = components(p)

    return render(request, 'map.html', {
        'bokeh_script': script,
        'bokeh_div': div,
    })


#######################################
# 2) get_drill_hole_data => intervals
#######################################
def get_drill_hole_data(request, hole_id):
    """
    Return JSON with intervals & element values for vertical lines in 3D.
    """
    try:
        hole = DrillHole.objects.get(hole_id=hole_id)

        # intervals
        intervals = list(DrillInterval.objects.filter(
            drill_hole=hole
        ).order_by('depth_from').values(
            'depth_from', 'depth_to', 'lithology'
        ))

        # element values
        elements = list(ElementValue.objects.filter(
            analysis__interval__drill_hole=hole
        ).values(
            'element',
            'value',
            'analysis__interval__depth_from',
            'analysis__interval__depth_to'
        ))

        data = {
            'hole_id': hole_id,
            'easting': hole.easting,
            'northing': hole.northing,
            'intervals': intervals,
            'elements': elements
        }
        return JsonResponse(data)
    except DrillHole.DoesNotExist:
        return JsonResponse({'error': f'Drill hole {hole_id} not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


#######################################
# 3) get_3d_model_for_hole => IDW volume
#######################################
def get_3d_model_for_hole(request, hole_id):
    """
    Returns JSON with { x, y, z, u } for a more 'land-like' 3D volume via IDW.
    - Uses a larger bounding box: +/-100m in XY around the hole
    - Uses a bigger grid: nx=30, ny=30, nz=20
    - Depth is from 0 to the max interval depth.
    - IDW radius is bigger (e.g. 999999) so we capture all intervals,
      or 120.0 if you want a radial limit.
    - If you want to see negative Z for downward, just invert the sign.
    """
    from django.db.models import Avg
    import numpy as np
    import pandas as pd
    from django.http import JsonResponse
    from .models import DrillHole, DrillInterval, ElementValue
    
    try:
        # 1) Fetch the hole
        hole = DrillHole.objects.get(hole_id=hole_id)
        
        # 2) Gather intervals => average U
        intervals_qs = DrillInterval.objects.filter(drill_hole=hole).order_by('depth_from')
        rows = []
        for interval in intervals_qs:
            u_qs = ElementValue.objects.filter(
                analysis__interval=interval,
                element='U'
            )
            if u_qs.exists():
                u_avg = u_qs.aggregate(Avg('value'))['value__avg'] or 0.0
            else:
                u_avg = 0.0
            
            mid = 0.5*(interval.depth_from + interval.depth_to)
            # If you prefer negative depth for "down", do `-mid` here
            rows.append({'depth_mid': mid, 'U_ppm': u_avg})
        
        df = pd.DataFrame(rows)
        if df.empty:
            return JsonResponse({'error': f"No intervals for hole {hole_id}"}, status=404)
        
        # 3) Define bounding box + grid
        e0, n0 = hole.easting, hole.northing
        min_depth = 0.0
        max_depth = df['depth_mid'].max()
        
        # A bigger bounding box so you see a "land-like" shape horizontally
        box_xy = 100.0  # +/- 100m
        nx, ny, nz = 30, 30, 20  # more refined grid
        xvals = np.linspace(e0 - box_xy, e0 + box_xy, nx)
        yvals = np.linspace(n0 - box_xy, n0 + box_xy, ny)
        zvals = np.linspace(min_depth, max_depth, nz)  # or invert sign if you want negative
        
        X, Y, Z = np.meshgrid(xvals, yvals, zvals, indexing='ij')
        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z.flatten()
        
        # 4) Prepare the intervals as "points" for IDW
        #    If you want negative downward, do:  depths = -df['depth_mid']
        #    Otherwise just keep them as is
        borehole_depths = df['depth_mid'].values
        borehole_u      = df['U_ppm'].values
        
        # 5) IDW Interpolation
        p = 2.0
        # Larger radius => basically includes all intervals
        radius = 999999.0  # or 120.0 if you want a limit
        Uf = np.zeros_like(Xf)
        
        for i in range(len(Xf)):
            dx = e0 - Xf[i]
            dy = n0 - Yf[i]
            # distance in 3D if you want to consider depth variation:
            dists = np.sqrt(dx**2 + dy**2 + (borehole_depths - Zf[i])**2)
            
            # optional radius filter
            mask = (dists < radius)
            if not np.any(mask):
                Uf[i] = 0.0
                continue
            
            dd = dists[mask]
            uu = borehole_u[mask]
            
            w = 1.0 / (dd**p + 1e-9)
            Uf[i] = np.sum(uu * w) / np.sum(w)
        
        # 6) Return as JSON
        data = {
            'x': Xf.tolist(),
            'y': Yf.tolist(),
            'z': Zf.tolist(),
            'u': Uf.tolist()
        }
        return JsonResponse(data)
    
    except DrillHole.DoesNotExist:
        return JsonResponse({'error': f"Hole {hole_id} not found"}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

from django.shortcuts import render
from .models import DrillHole, DrillInterval, ElementValue
from django.db.models import Avg, Max, Count
import pandas as pd

# ... (Your existing functions: process_aura_file, upload_excel, generate_report_pdf, etc.) ...

# --- Functions for the index view (using Chart.js) ---

def calculate_summary_stats():
    """Calculates summary statistics for the dashboard."""
    total_drill_holes = DrillHole.objects.count()
    average_uranium = ElementValue.objects.filter(element='U').aggregate(Avg('value'))['value__avg'] or 0
    granite_intervals = DrillInterval.objects.filter(lithology__icontains='granite').count()
    max_uranium = ElementValue.objects.filter(element='U').aggregate(Max('value'))['value__max'] or 0
    return total_drill_holes, average_uranium, granite_intervals, max_uranium

def prepare_data_for_charts_index():
    """Prepares data for Chart.js charts on the index page."""
    # ... (Same data preparation logic as before) ...
    intervals = DrillInterval.objects.select_related('drill_hole').prefetch_related('chemical_analyses__element_values').all()
    data_rows = []
    for interval in intervals:
        hole = interval.drill_hole
        avg_depth = (interval.depth_from + interval.depth_to) / 2.0
        for analysis in interval.chemical_analyses.all():
            element_map = {ev.element.upper(): ev.value for ev in analysis.element_values.all()}
            row_dict = {
                'hole_id': hole.hole_id,
                'depth': avg_depth,
                'lithology': interval.lithology.lower() if interval.lithology else "unknown",
                'U': element_map.get('U', 0),
            }
            data_rows.append(row_dict)

    df = pd.DataFrame(data_rows)

    # Convert relevant columns to numeric, handling non-numeric values
    df['U'] = pd.to_numeric(df['U'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df.dropna(subset=['U', 'depth'], inplace=True)

    return df

# --- New index view ---
from django.core.serializers.json import DjangoJSONEncoder
import json

def index(request):
    # Data aggregations
    total_drill_holes = DrillHole.objects.count()
    average_uranium = ElementValue.objects.filter(element='U').aggregate(Avg('value'))['value__avg'] or 0
    granite_intervals = DrillInterval.objects.filter(lithology__icontains='granite').count()
    max_uranium = ElementValue.objects.filter(element='U').aggregate(Max('value'))['value__max'] or 0
    
    # Prepare data for charts
    intervals = DrillInterval.objects.select_related('drill_hole').prefetch_related('chemical_analyses__element_values').all()
    data_rows = []
    for interval in intervals:
        hole = interval.drill_hole
        avg_depth = (interval.depth_from + interval.depth_to) / 2.0
        for analysis in interval.chemical_analyses.all():
            element_map = {ev.element.upper(): ev.value for ev in analysis.element_values.all()}
            row_dict = {
                'hole_id': hole.hole_id,
                'depth': float(avg_depth),  # Convert to float
                'lithology': interval.lithology.lower() if interval.lithology else "unknown",
                'U': float(element_map.get('U', 0)),  # Convert to float
            }
            data_rows.append(row_dict)
    
    df = pd.DataFrame(data_rows)
    
    # Prepare data for Chart.js charts and convert to JSON-safe types
    lithology_counts = df['lithology'].value_counts()
    uranium_by_depth = df.groupby('depth')['U'].mean().reset_index()
    
    context = {
        'total_drill_holes': total_drill_holes,
        'average_uranium': round(float(average_uranium), 2),
        'granite_intervals': granite_intervals,
        'max_uranium': round(float(max_uranium), 2),
        'lithology_labels': json.dumps(list(lithology_counts.index)),
        'lithology_counts': json.dumps([float(x) for x in lithology_counts.values]),
        'uranium_depth_labels': json.dumps([float(x) for x in uranium_by_depth['depth']]),
        'uranium_depth_data': json.dumps([float(x) for x in uranium_by_depth['U']]),
    }
    
    return render(request, 'index.html', context)

def intro(request):
    return render(request, 'intro.html')