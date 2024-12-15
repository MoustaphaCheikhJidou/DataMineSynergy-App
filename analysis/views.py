import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
from django.core.exceptions import ValidationError
from django.apps import apps
import tempfile
import re
import logging
import os
from django.db import models

# Import des modèles
from .models import DrillHole, DrillInterval, ChemicalAnalysis, ElementValue

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
            df_geology = pd.read_excel(file_path, sheet_name="DHGeology")
            df_assays = pd.read_excel(file_path, sheet_name="DHAssays")
            logger.info(f"Feuilles chargées avec succès. Géologie: {len(df_geology)} lignes, Analyses: {len(df_assays)} lignes")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des feuilles Excel: {str(e)}")
            raise ValueError(f"Erreur lors de la lecture des feuilles Excel. Assurez-vous que les feuilles 'DHGeology' et 'DHAssays' existent. Erreur: {str(e)}")

        # Standardiser les noms de colonnes
        column_mapping = {
            'HOLEID': 'HoleID',
            'Hole_ID': 'HoleID',
            'holeid': 'HoleID',
            'HOLE_ID': 'HoleID',
            'PROJECT': 'Project',
            'project': 'Project',
            'PROSPECT': 'Prospect',
            'prospect': 'Prospect',
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
        df_geology = df_geology.rename(columns=column_mapping)
        df_assays = df_assays.rename(columns=column_mapping)

        # Vérifier les colonnes requises
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
        df_assays = df_assays[comparison_assays]

        # Suppression des NaN uniquement dans la colonne 'U_ppm'
        if 'U_ppm' in df_assays.columns:
            df_assays = df_assays[df_assays['U_ppm'].notna()]

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
        df_assays['ID'] = df_assays['HoleID'].astype(str) + df_assays['DepthFrom'].astype(str) + df_assays['DepthTo'].astype(str)

        # Sélectionner les colonnes d'analyses chimiques
        assay_cols = ['ID', 'Project', 'HoleID', 'DepthFrom', 'DepthTo'] + [col for col in df_assays.columns if any(element in col for element in ['_pct', '_ppm', '_ppb'])]
        df_assays = df_assays[assay_cols]

        # Réinitialiser les indices
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
            unique_holes = df_merged[['HoleID', 'Project', 'Prospect']].drop_duplicates()
            logger.info(f"Nombre de trous uniques: {len(unique_holes)}")
            
            for index, row in unique_holes.iterrows():
                try:
                    DrillHole.objects.create(
                        hole_id=str(row.HoleID),  # Utiliser la notation point
                        project=str(row.Project),
                        prospect=str(row.Prospect) if pd.notna(row.Prospect) else None
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
                            if pd.notna(value) and value != 0:
                                try:
                                    element = col.split('_')[0]
                                    unit = col.split('_')[1]
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
        
        return redirect('upload_excel')
    
    return render(request, 'upload.html')
