import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DrillHole',
            fields=[
                ('hole_id', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('project', models.CharField(blank=True, max_length=255, null=True)),
                ('prospect', models.CharField(blank=True, max_length=255, null=True)),
                ('easting', models.FloatField(blank=True, null=True)),
                ('northing', models.FloatField(blank=True, null=True)),
            ],
            options={
                'indexes': [models.Index(fields=['project'], name='analysis_dr_project_ac73c8_idx'), models.Index(fields=['prospect'], name='analysis_dr_prospec_66db2b_idx')],
            },
        ),
        migrations.CreateModel(
            name='DrillInterval',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('depth_from', models.FloatField()),
                ('depth_to', models.FloatField()),
                ('lithology', models.CharField(blank=True, max_length=255, null=True)),
                ('drill_hole', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='intervals', to='analysis.drillhole')),
            ],
        ),
        migrations.CreateModel(
            name='ChemicalAnalysis',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('interval', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chemical_analyses', to='analysis.drillinterval')),
            ],
        ),
        migrations.CreateModel(
            name='ElementValue',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('element', models.CharField(max_length=20)),
                ('value', models.FloatField()),
                ('unit', models.CharField(max_length=10)),
                ('analysis', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='element_values', to='analysis.chemicalanalysis')),
            ],
        ),
        migrations.AddIndex(
            model_name='drillinterval',
            index=models.Index(fields=['drill_hole', 'depth_from', 'depth_to'], name='analysis_dr_drill_h_6ca3c9_idx'),
        ),
        migrations.AddConstraint(
            model_name='drillinterval',
            constraint=models.CheckConstraint(check=models.Q(('depth_to__gt', models.F('depth_from'))), name='depth_check'),
        ),
        migrations.AddIndex(
            model_name='chemicalanalysis',
            index=models.Index(fields=['interval'], name='analysis_ch_interva_bb4eec_idx'),
        ),
        migrations.AddIndex(
            model_name='elementvalue',
            index=models.Index(fields=['element'], name='analysis_el_element_9cadc6_idx'),
        ),
        migrations.AddIndex(
            model_name='elementvalue',
            index=models.Index(fields=['analysis'], name='analysis_el_analysi_5da6ad_idx'),
        ),
        migrations.AddConstraint(
            model_name='elementvalue',
            constraint=models.CheckConstraint(check=models.Q(('value__gte', 0)), name='non_negative_value'),
        ),
    ]
