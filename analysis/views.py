import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
from django.core.exceptions import ValidationError
from django.apps import apps
import tempfile
import re
import logging

from django.db import models

# Define your models here or ensure they are imported correctly
from .models import (
    Collars,
    DHDrillType,
    DHSurvey,
    DHSamples,
    DHGeology,
    DHScint,
    DHMagSus,
    DHAssays,
)

# Configure logger
logger = logging.getLogger(__name__)

# your_app/views.py

COLUMN_MAPPING = {
    'Collars': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'HoleID': 'holeid',
        'Prospect': 'prospect',
        'Projection': 'projection',
        'Easting': 'easting',
        'Northing': 'northing',
        'SurveyMethod': 'survey_method',
        'RL': 'rl',
        'PlottingRL': 'plotting_rl',
        'Dip': 'dip',
        'MagDeclination': 'mag_declination',
        'AzimMag': 'azim_mag',
        'AzimUTM': 'azim_utm',
        'TotalDepth': 'total_depth',
        'LoggedBy': 'logged_by',
        'TenementName': 'tenement_name',
        'TenementNo': 'tenement_no',
        'Comments': 'comments',
        'WaterDepth': 'water_depth',
        'WaterComments': 'water_comments',
    },
    'DHDrillType': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'HoleID': 'hole_id',
        'DepthFrom': 'depth_from',
        'DepthTo': 'depth_to',
        'DrillType': 'drill_type',
        'HoleSize': 'hole_size',
        'DrillRig': 'drill_rig',
        'CasingDepth': 'casing_depth',
        'CasingType': 'casing_type',
        'DrillContractor': 'drill_contractor',
        'DateStarted': 'date_started',
        'DateCompleted': 'date_completed',
        'CoreBoxes': 'core_boxes',
        'Comments': 'comments',
    },
    'DHSurvey': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'HoleID': 'hole_id',
        'Depth': 'depth',
        'Dip': 'dip',
        'AzimMag': 'azim_mag',
        'AzimUTM': 'azim_utm',
        'DateSurveyed': 'date_surveyed',
        'DHSurveyInstrument': 'dh_survey_instrument',
        'Valid': 'valid',
        'MagneticField_nT': 'magnetic_field_n_t',
        'SurveyedBy': 'surveyed_by',
        'Comments': 'comments',
    },
    'DHSamples': {
        'Project': 'project',
        'HoleID': 'hole_id',
        'SampleID': 'sample_id',
        'DepthFrom': 'depth_from',
        'DepthTo': 'depth_to',
        'Interval': 'interval',
        'SampType': 'samp_type',
        'SampleSubType': 'sample_sub_type',
        'SampleWeightKg': 'sample_weight_kg',
        'ChkType': 'chk_type',
        'ParentID': 'parent_id',
        'StandardID': 'standard_id',
        'SampledBy': 'sampled_by',
        'DateSampled': 'date_sampled',
        'Comments': 'comments',
    },
    'DHGeology': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'Prospect': 'prospect',
        'HoleID': 'hole_id',
        'DepthFrom': 'depth_from',
        'DepthTo': 'depth_to',
        'Flag': 'flag',
        'Regolith': 'regolith',
        'Oxidation': 'oxidation',
        'Moisture': 'moisture',
        'Weathering': 'weathering',
        'Hue': 'hue',
        'Colour1': 'colour1',
        'Colour2': 'colour2',
        'Classifiers': 'classifiers',
        'Grainsize': 'grainsize',
        'Lithology1': 'lithology1',
        'Lith1_pct': 'lith1_pct',
        'Lithology2': 'lithology2',
        'Lith2_pct': 'lith2_pct',
        'Texture1': 'texture1',
        'Texture2': 'texture2',
        'Texture3': 'texture3',
        'RockMin1': 'rock_min1',
        'RockMin1_pct': 'rock_min1_pct',
        'RockMin2': 'rock_min2',
        'RockMin2_pct': 'rock_min2_pct',
        'RockMin3': 'rock_min3',
        'RockMin3_pct': 'rock_min3_pct',
        'RockMin4': 'rock_min4',
        'RockMin4_pct': 'rock_min4_pct',
        'RockMin5': 'rock_min5',
        'RockMin5_pct': 'rock_min5_pct',
        'LoggedBy': 'logged_by',
        'DateLogged': 'date_logged',
        'Comments': 'comments',
    },
    'DHScint': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'HoleID': 'hole_id',
        'DepthFrom': 'depth_from',
        'DepthTo': 'depth_to',
        'Average_cps': 'average_cps',
        'Average_uS/h': 'average_us_h',
        'CB_eTh': 'cb_e_th',
        'CB_eU': 'cb_e_u',
        'CB_K': 'cb_k',
        'Dip_core': 'dip_core',
        'GB_eT': 'gb_e_t',
        'GB_eU': 'gb_e_u',
        'GB_K': 'gb_k',
        'Scintillometer': 'scintillometer',
        'Scintillometer_cps': 'scintillometer_cps',
        'Scintillometer1': 'scintillometer1',
        'Scintillometer1_cps': 'scintillometer1_cps',
        'Scintillometer2': 'scintillometer2',
        'Scintillometer2_cps': 'scintillometer2_cps',
        'Scintillometer3': 'scintillometer3',
        'Scintillometer3_cps': 'scintillometer3_cps',
        'ScintUnit': 'scint_unit',
        'uR/h': 'ur_h',
        'Dip': 'dip',
        'LoggedBy': 'logged_by',
        'DateLogged': 'date_logged',
        'Comments': 'comments',
    },
    'DHMagSus': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'HoleID': 'hole_id',
        'DepthFrom': 'depth_from',
        'DepthTo': 'depth_to',
        'MagSusInstrument': 'mag_sus_instrument',
        'MagSusUnit': 'mag_sus_unit',
        'MagSus1': 'mag_sus1',
        'MagSus2': 'mag_sus2',
        'MagSus3': 'mag_sus3',
        'MagSusAverage': 'mag_sus_average',
        'LoggedBy': 'logged_by',
        'DateLogged': 'date_logged',
    },
    'DHAssays': {
        'Company': 'company',
        'Country': 'country',
        'Ownership': 'ownership',
        'Project': 'project',
        'Prospect': 'prospect',
        'HoleID': 'hole_id',
        'SampleID': 'sample_id',
        'DepthFrom': 'depth_from',
        'DepthTo': 'depth_to',
        'Interval': 'interval',
        'SampType': 'samp_type',
        'SampleSubType': 'sample_sub_type',
        'SampledBy': 'sampled_by',
        'DateSampled': 'date_sampled',
        'Comments': 'comments',
        'Au_ppm': 'au_ppm',
        '-75µm_pct': '_75um_pct',
        'Ag_ppm': 'ag_ppm',
        'Al_pct': 'al_pct',
        'Al2O3_pct': 'al2o3_pct',
        'As_ppm': 'as_ppm',
        'B_ppm': 'b_ppm',
        'Ba_ppm': 'ba_ppm',
        'BaO_pct': 'bao_pct',
        'Be_ppm': 'be_ppm',
        'Bi_pct': 'bi_pct',
        'Bi_ppm': 'bi_ppm',
        'Br_ppm': 'br_ppm',
        'C_pct': 'c_pct',
        'Ca_pct': 'ca_pct',
        'CaO_pct': 'cao_pct',
        'Cd_ppm': 'cd_ppm',
        'Ce_ppm': 'ce_ppm',
        'Cl_pct': 'cl_pct',
        'Co_ppm': 'co_ppm',
        'CO2_pct': 'co2_pct',
        'Cr_ppm': 'cr_ppm',
        'Cr2O3_pct': 'cr2o3_pct',
        'Cs_ppm': 'cs_ppm',
        'Cu_pct': 'cu_pct',
        'Cu_ppm': 'cu_ppm',
        'CuO_pct': 'cuo_pct',
        'Dy_ppm': 'dy_ppm',
        'Er_ppm': 'er_ppm',
        'Eu_ppm': 'eu_ppm',
        'Fe_pct': 'fe_pct',
        'Fe2O3_pct': 'fe2o3_pct',
        'Ga_ppm': 'ga_ppm',
        'Gd_ppm': 'gd_ppm',
        'Ge_ppm': 'ge_ppm',
        'Hf_ppm': 'hf_ppm',
        'Hg_ppm': 'hg_ppm',
        'Ho_ppm': 'ho_ppm',
        'In_ppm': 'in_ppm',
        'Ir_ppb': 'ir_ppb',
        'K_pct': 'k_pct',
        'K2O_pct': 'k2o_pct',
        'La_ppm': 'la_ppm',
        'Li_ppm': 'li_ppm',
        'LOI_pct': 'loi_pct',
        'Lu_ppm': 'lu_ppm',
        'Mg_pct': 'mg_pct',
        'MgO_pct': 'mgo_pct',
        'Mn_ppm': 'mn_ppm',
        'MnO_pct': 'mno_pct',
        'Mo_ppm': 'mo_ppm',
        'Na_pct': 'na_pct',
        'Na2O_pct': 'na2o_pct',
        'Nb_ppm': 'nb_ppm',
        'Nd_ppm': 'nd_ppm',
        'Ni_ppm': 'ni_ppm',
        'P_ppm': 'p_ppm',
        'P2O5_pct': 'p2o5_pct',
        'Pass2mm_pct': 'pass2mm_pct',
        'Pass75um_pct': 'pass75um_pct',
        'Pb_ppm': 'pb_ppm',
        'PbO_pct': 'pbo_pct',
        'Pd_ppm': 'pd_ppm',
        'Pr_ppm': 'pr_ppm',
        'Pt_ppm': 'pt_ppm',
        'Rb_ppm': 'rb_ppm',
        'Re_ppm': 're_ppm',
        'S_pct': 's_pct',
        'Sb_ppm': 'sb_ppm',
        'Sc_ppm': 'sc_ppm',
        'Se_ppm': 'se_ppm',
        'SiO2_pct': 'sio2_pct',
        'Sm_ppm': 'sm_ppm',
        'Sn_ppm': 'sn_ppm',
        'Sr_ppm': 'sr_ppm',
        'SrO_pct': 'sro_pct',
        'Ta_ppm': 'ta_ppm',
        'Tb_ppm': 'tb_ppm',
        'Te_ppm': 'te_ppm',
        'Th_ppm': 'th_ppm',
        'Ti_pct': 'ti_pct',
        'TiO2_pct': 'tio2_pct',
        'Tl_ppm': 'tl_ppm',
        'Tm_ppm': 'tm_ppm',
        'U_ppm': 'u_ppm',
        'V_ppm': 'v_ppm',
        'V2O5_pct': 'v2o5_pct',
        'W_ppm': 'w_ppm',
        'Y_ppm': 'y_ppm',
        'Yb_ppm': 'yb_ppm',
        'Zn_ppm': 'zn_ppm',
        'Zr_ppm': 'zr_ppm',
        'BatchNo': 'batch_no',
    },
}
def sanitize_field_name(sheet_name, column_name):
    """Map Excel column names to Django model field names."""
    # Retrieve the mapping for the given sheet
    sheet_mapping = COLUMN_MAPPING.get(sheet_name, {})
    # Return the mapped field name if exists, else apply default sanitization
    return sheet_mapping.get(column_name, default_sanitize(column_name))

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

def upload_excel(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            messages.error(request, 'No file was uploaded.')
            return redirect('upload_excel')

        # Validate file extension
        if not uploaded_file.name.endswith(('.xlsx', '.xls')):
            messages.error(request, 'Unsupported file type. Please upload an Excel file.')
            return redirect('upload_excel')

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            insert_data_from_excel(tmp_path)
            messages.success(request, 'Excel file imported successfully.')
        except ValidationError as ve:
            messages.error(request, f'Validation Error: {ve}')
            logger.error(f'Validation Error: {ve}')
        except Exception as e:
            messages.error(request, f'Error importing data: {e}')
            logger.exception('Unexpected error during Excel import.')

        return redirect('upload_excel')

    return render(request, 'upload.html')

# your_app/views.py
def insert_data_from_excel(excel_file_path):
    """
    Inserts data from an Excel file into the corresponding Django models.

    Args:
        excel_file_path (str): The path to the Excel file.
    """
    logger.info(f"Starting import from Excel file: {excel_file_path}")
    # Read the Excel file
    try:
        xl = pd.ExcelFile(excel_file_path)
        logger.info("Excel file read successfully.")
    except Exception as e:
        logger.exception(f"Failed to read Excel file: {e}")
        raise ValidationError("Invalid Excel file.")

    # Define required sheets based on your models
    required_sheets = [
        'Collars',
        'DHDrillType',
        'DHSurvey',
        'DHSamples',
        'DHGeology',
        'DHScint',
        'DHMagSus',
        'DHAssays'
    ]

    # Check for missing sheets
    missing_sheets = [sheet for sheet in required_sheets if sheet not in xl.sheet_names]
    if missing_sheets:
        logger.error(f"Missing required sheets: {', '.join(missing_sheets)}")
        raise ValidationError(f"Missing required sheets: {', '.join(missing_sheets)}")

    for sheet_name in required_sheets:
        logger.info(f"Processing sheet: {sheet_name}")
        df = xl.parse(sheet_name)

        # Sanitize sheet name to get the model name
        model_name = sanitize_sheet_name(sheet_name)

        # Get the model class
        try:
            model = apps.get_model('analytics', model_name)  # Ensure 'analytics' is your app name
            logger.info(f"Retrieved model: {model_name}")
        except LookupError:
            logger.error(f"No model found for sheet '{sheet_name}'.")
            raise ValidationError(f"No model found for sheet '{sheet_name}'.")

        # Prepare list to hold model instances
        instances = []

        # Iterate over DataFrame rows as (index, Series) pairs
        for index, row in df.iterrows():
            data = {}
            for column in df.columns:
                field_name = sanitize_field_name(sheet_name, column)
                value = row[column]

                # Handle NaN values
                if pd.isna(value):
                    value = None

                # Optional: Add data type conversions based on field type
                try:
                    field = model._meta.get_field(field_name)
                    if isinstance(field, models.DateField):
                        if isinstance(value, pd.Timestamp):
                            value = value.date()
                        elif isinstance(value, str):
                            value = pd.to_datetime(value, errors='coerce').date()
                        else:
                            value = None
                    elif isinstance(field, models.BooleanField):
                        if isinstance(value, str):
                            value = value.lower() in ['true', '1', 'yes']
                        elif isinstance(value, (int, float)):
                            value = bool(value)
                        else:
                            value = None
                    elif isinstance(field, models.FloatField):
                        if isinstance(value, str):
                            try:
                                value = float(value)
                            except ValueError:
                                value = None
                        else:
                            value = float(value) if pd.notna(value) else None
                    elif isinstance(field, models.IntegerField):
                        if isinstance(value, str):
                            try:
                                value = int(value)
                            except ValueError:
                                value = None
                        else:
                            value = int(value) if pd.notna(value) else None
                    elif isinstance(field, models.CharField) or isinstance(field, models.TextField):
                        value = str(value).strip() if pd.notna(value) else ''
                    # Add more type conversions if needed
                except (ValueError, TypeError) as e:
                    logger.warning(f"Row {index + 2}: Error converting field '{field_name}' with value '{value}': {e}")
                    value = None  # Or handle the error as needed

                data[field_name] = value

            # Create a model instance without saving to the database
            try:
                instance = model(**data)
                instances.append(instance)
            except Exception as e:
                logger.error(f"Error creating instance for row {index + 2} in sheet '{sheet_name}': {e}")
                raise ValidationError(f"Error creating instance for row {index + 2} in sheet '{sheet_name}': {e}")

        # Bulk create instances within a transaction
        try:
            with transaction.atomic():
                created_count = model.objects.bulk_create(instances, batch_size=1000)
                logger.info(f"Imported {len(created_count)} records into {model_name}.")
        except Exception as e:
            logger.exception(f"Error bulk creating records for {model_name}: {e}")
            raise e



