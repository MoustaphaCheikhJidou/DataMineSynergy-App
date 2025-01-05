import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
from django.core.exceptions import ValidationError
from django.apps import apps
import tempfile
import re
import logging
<<<<<<< HEAD
import os
from django.db import models

# Import des modèles
from .models import DrillHole, DrillInterval, ChemicalAnalysis, ElementValue

=======
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
>>>>>>> front
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
<<<<<<< HEAD
        
        # Charger les feuilles dans des DataFrames
        logger.info("Lecture des feuilles Excel...")
        try:
            df_geology = pd.read_excel(file_path, sheet_name="DHGeology")
            df_assays = pd.read_excel(file_path, sheet_name="DHAssays")
            logger.info(f"Feuilles chargées avec succès. Géologie: {len(df_geology)} lignes, Analyses: {len(df_assays)} lignes")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des feuilles Excel: {str(e)}")
            raise ValueError(f"Erreur lors de la lecture des feuilles Excel. Assurez-vous que les feuilles 'DHGeology' et 'DHAssays' existent. Erreur: {str(e)}")

        # Standardiser les noms de colonnes
        column_mapping = {
=======

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
>>>>>>> front
            'HOLEID': 'HoleID',
            'Hole_ID': 'HoleID',
            'holeid': 'HoleID',
            'HOLE_ID': 'HoleID',
            'PROJECT': 'Project',
            'project': 'Project',
            'PROSPECT': 'Prospect',
            'prospect': 'Prospect',
<<<<<<< HEAD
            'DEPTHFROM': 'DepthFrom',
            'Depth_From': 'DepthFrom',
            'depthfrom': 'DepthFrom',
            'DEPTH_FROM': 'DepthFrom',
            'DEPTHTO': 'DepthTo',
            'Depth_To': 'DepthTo',
            'depthto': 'DepthTo',
            'DEPTH_TO': 'DepthTo',
            'LITHOLOGY1': 'Lithology1',
            'lithology1': 'Lithology1',
            'Lithology_1': 'Lithology1'
        }

        # Appliquer le mapping aux deux DataFrames
=======
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
>>>>>>> front
        df_geology = df_geology.rename(columns=column_mapping)
        df_assays = df_assays.rename(columns=column_mapping)

        # Vérifier les colonnes requises
<<<<<<< HEAD
        required_geology_columns = ['HoleID', 'Project', 'DepthFrom', 'DepthTo', 'Lithology1']
        required_assays_columns = ['HoleID', 'Project', 'DepthFrom', 'DepthTo']

        # Vérifier si toutes les colonnes requises sont présentes
        missing_geology = [col for col in required_geology_columns if col not in df_geology.columns]
        missing_assays = [col for col in required_assays_columns if col not in df_assays.columns]

        if missing_geology or missing_assays:
            raise ValueError(f"Colonnes manquantes - Géologie: {missing_geology}, Analyses: {missing_assays}")

        # Sélectionner les colonnes nécessaires de la géologie
        df_geology = df_geology[['Project', 'Prospect', 'HoleID', 'DepthFrom', 'DepthTo', 'Lithology1']]

        # Comparer les colonnes 'HoleID' dans les deux DataFrames
        comparison = df_geology['HoleID'].isin(df_assays['HoleID'])
        df_geology = df_geology[comparison]

        # Comparer les HoleID dans df_assays et df_geology
        comparison_assays = df_assays['HoleID'].isin(df_geology['HoleID'])
=======
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
>>>>>>> front
        df_assays = df_assays[comparison_assays]

        # Suppression des NaN uniquement dans la colonne 'U_ppm'
        if 'U_ppm' in df_assays.columns:
            df_assays = df_assays[df_assays['U_ppm'].notna()]

<<<<<<< HEAD
        # Liste des colonnes à exclure
        exclude_columns = ['DepthFrom', 'DepthTo', 'Interval', 'Comments']

        # Identification des colonnes numériques à traiter
        numeric_columns = df_assays.select_dtypes(include='number').columns
        columns_to_process = [col for col in numeric_columns if col not in exclude_columns]

        # Remplacement des valeurs négatives par 0 dans les colonnes numériques sélectionnées
        df_assays[columns_to_process] = df_assays[columns_to_process].clip(lower=0)

        # Remplacement des NaN restants par 0
        df_assays = df_assays.fillna(0)

        # Créer une colonne ID en combinant HoleID, DepthFrom et DepthTo
        df_geology['ID'] = df_geology['HoleID'].astype(str) + df_geology['DepthFrom'].astype(str) + df_geology['DepthTo'].astype(str)
=======
        # Créer une colonne ID en combinant HoleID, DepthFrom et DepthTo
        df_merged_collars_geology['ID'] = df_merged_collars_geology['HoleID'].astype(str) + df_merged_collars_geology['DepthFrom'].astype(str) + df_merged_collars_geology['DepthTo'].astype(str)
>>>>>>> front
        df_assays['ID'] = df_assays['HoleID'].astype(str) + df_assays['DepthFrom'].astype(str) + df_assays['DepthTo'].astype(str)

        # Sélectionner les colonnes d'analyses chimiques
        assay_cols = ['ID', 'Project', 'HoleID', 'DepthFrom', 'DepthTo'] + [col for col in df_assays.columns if any(element in col for element in ['_pct', '_ppm', '_ppb'])]
        df_assays = df_assays[assay_cols]

        # Réinitialiser les indices
<<<<<<< HEAD
        df_geology = df_geology.reset_index(drop=True)
        df_assays = df_assays.reset_index(drop=True)

        # Trouver les ID communs
        common_ids = df_geology['ID'].isin(df_assays['ID'])
        df_geology_common = df_geology[common_ids]

        common_ids_assays = df_assays['ID'].isin(df_geology_common['ID'])
        df_assays_common = df_assays[common_ids_assays]

        # Identifier et supprimer les colonnes communes de df_geology_common
        common_columns = set(df_geology_common.columns).intersection(set(df_assays_common.columns)) - {'ID'}
        df_geology_common = df_geology_common.drop(columns=common_columns)

        # Effectuer la fusion
        df_merged = pd.merge(df_assays_common, df_geology_common, on='ID', how='left')

        # Identifier et supprimer les colonnes avec uniquement des zéros
        cols_to_exclude = []
        for col in df_merged.columns:
            if df_merged[col].unique().tolist() == [0]:
                cols_to_exclude.append(col)
        df_merged = df_merged.drop(cols_to_exclude, axis=1)
=======
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
>>>>>>> front

        # Convertir toutes les valeurs de Lithology1 en minuscules
        df_merged['Lithology1'] = df_merged['Lithology1'].str.lower()

        # Afficher les informations sur les données fusionnées
        logger.info(f"Colonnes dans df_merged: {df_merged.columns.tolist()}")
        logger.info(f"Nombre de lignes dans df_merged: {len(df_merged)}")

        # Sauvegarder les données dans la base de données
        with transaction.atomic():
            # Supprimer les anciennes données
            DrillHole.objects.all().delete()
            
            # Créer les trous de forage (approche modifiée)
<<<<<<< HEAD
            unique_holes = df_merged[['HoleID', 'Project', 'Prospect']].drop_duplicates()
=======
            unique_holes = df_merged[['HoleID', 'Project', 'Prospect', 'Easting', 'Northing']].drop_duplicates()
>>>>>>> front
            logger.info(f"Nombre de trous uniques: {len(unique_holes)}")
            
            for index, row in unique_holes.iterrows():
                try:
                    DrillHole.objects.create(
<<<<<<< HEAD
                        hole_id=str(row.HoleID),  # Utiliser la notation point
                        project=str(row.Project),
                        prospect=str(row.Prospect) if pd.notna(row.Prospect) else None
=======
                        hole_id=str(row.HoleID),
                        project=str(row.Project),
                        prospect=str(row.Prospect) if pd.notna(row.Prospect) else None,
                        easting=float(row.Easting) if pd.notna(row.Easting) else None,  # Handle missing values
                        northing=float(row.Northing) if pd.notna(row.Northing) else None   # Handle missing values
>>>>>>> front
                    )
                except Exception as e:
                    logger.error(f"Erreur lors de la création du trou {row.HoleID}: {str(e)}")
                    raise

            # Créer les intervalles et analyses chimiques
            for index, row in df_merged.iterrows():
                try:
                    drill_hole = DrillHole.objects.get(hole_id=str(row.HoleID))
                    
                    interval = DrillInterval.objects.create(
                        drill_hole=drill_hole,
                        depth_from=float(row.DepthFrom),
                        depth_to=float(row.DepthTo),
                        lithology=str(row.Lithology1)
                    )

                    # Créer l'analyse chimique
                    chemical_analysis = ChemicalAnalysis.objects.create(
                        interval=interval
                    )

                    # Ajouter les valeurs des éléments
                    for col in df_merged.columns:
                        if any(suffix in col.lower() for suffix in ['_ppm', '_ppb', '_pct']):
                            value = row[col]
<<<<<<< HEAD
                            if pd.notna(value) and value != 0:
                                try:
                                    element = col.split('_')[0]
                                    unit = col.split('_')[1]
=======
                            if pd.notna(value):
                                try:
                                    element = col.split('_')[0]
                                    unit = col.split('_')[1]
                                    # Ensure the value is non-negative
                                    value = max(0.0, float(value))  # Set negative values to 0
>>>>>>> front
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
<<<<<<< HEAD

=======
    
@login_required
>>>>>>> front
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
        
<<<<<<< HEAD
        return redirect('upload_excel')
    
    return render(request, 'upload.html')
=======
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
    major_elements_plot = plot_major_elements_vs_si02(df_granite)
    trace_elements_plot = plot_trace_elements_vs_si02(df_granite)
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
        file_html(major_elements_plot, CDN, "Major Elements vs SiO2"),
        file_html(trace_elements_plot, CDN, "Trace Elements vs SiO2"),
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

def plot_major_elements_vs_si02(df):
    """
    Grid of major elements vs. SIO2, each in a separate small figure.
    Expect columns: SIO2, Al2O3, CaO, FEO, MgO, Na2O, K2O, TiO2
    """
    # Print available columns for debugging
    print("Available columns:", df.columns.tolist())
    print("Number of rows:", len(df))
    
    if df.empty:
        return Div(text="DataFrame is empty")
    
    if 'SIO2' not in df.columns:
        return Div(text="SIO2 column not found. Available columns: " + ", ".join(df.columns))

    # Define expected column names and their alternates
    major_elements = {
        'AL2O3': ['AL2O3', 'Al2O3', 'al2o3'],
        'CAO': ['CAO', 'CaO', 'cao'],
        'FEO': ['FEO', 'FeO', 'feo'],
        'MGO': ['MGO', 'MgO', 'mgo'],
        'NA2O': ['NA2O', 'Na2O', 'na2o'],
        'K2O': ['K2O', 'k2o'],
        'TIO2': ['TIO2', 'TiO2', 'tio2']
    }
    
    plots = []
    
    for elem_key, elem_variants in major_elements.items():
        # Find the first matching column name variant
        matching_col = next((col for col in elem_variants if col in df.columns), None)
        
        if matching_col:
            # Print data range for debugging
            print(f"{matching_col} range:", df[matching_col].min(), "to", df[matching_col].max())
            
            p = figure(width=350, height=300,
                      title=f"{elem_key} vs SiO2",
                      x_axis_label="SiO2 (%)",
                      y_axis_label=f"{elem_key} (%)",
                      tools="pan,box_zoom,reset,hover,save")
            
            source = ColumnDataSource(df)
            p.scatter(x='SIO2', y=matching_col, source=source, 
                    size=6, color="blue", alpha=0.6)
            
            p.hover.tooltips = [(elem_key, f"@{matching_col}"), ("SiO2", "@SIO2")]
            plots.append(p)
        else:
            print(f"No matching column found for {elem_key}")

    if not plots:
        return Div(text="No valid major element columns found for plotting")

    # Arrange plots in rows of 2
    rows = []
    for i in range(0, len(plots), 2):
        row_plots = plots[i:i+2]
        rows.append(row(*row_plots))
    

    return column(*rows)



def plot_trace_elements_vs_si02(df):
    """
    Similar approach but for trace elements. Expect columns [Ba, Nb, Rb, Sr, Zn, Zr] etc.
    """
    if df.empty or 'SIO2' not in df.columns:
        return Div(text="No data for Trace Elements vs SiO2 plot (SIO2 missing).")

    trace_elements = ['BA', 'NB', 'RB', 'SR', 'ZN', 'ZR']
    plots = []
    for elem in trace_elements:
        if elem in df.columns:
            p = figure(width=350, height=300,
                       title=f"{elem} vs SiO2",
                       x_axis_label="SiO2 (%)",
                       y_axis_label=f"{elem} (ppm)",
                       tools="pan,box_zoom,reset,hover,save")
            p.scatter(x='SIO2', y=elem, source=ColumnDataSource(df), size=6, color="green", alpha=0.5)
            p.hover.tooltips = [(f"{elem}", f"@{elem}"), ("SiO2", "@SIO2")]
            plots.append(p)

    rowed = []
    for i in range(0, len(plots), 2):
        rowed.append(row(*plots[i:i+2]))
    return column(*rowed)

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
    tab_major_elements = TabPanel(child=plot_major_elements_vs_si02(df_granite), 
                                  title="Major vs SiO2")

    # (e) Trace elements vs. SiO2
    tab_trace_elements = TabPanel(child=plot_trace_elements_vs_si02(df_granite), 
                                  title="Trace vs SiO2")

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
        tab_major_elements,
        tab_trace_elements,
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
>>>>>>> front
