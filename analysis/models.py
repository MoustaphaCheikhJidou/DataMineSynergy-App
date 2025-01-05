# excel_importer/models.py

from django.db import models



class DrillHole(models.Model):
    """Table principale des trous de forage"""
    hole_id = models.CharField(max_length=50, primary_key=True)
    project = models.CharField(max_length=255, null=True, blank=True)
    prospect = models.CharField(max_length=255, null=True, blank=True)
    easting = models.FloatField(null=True, blank=True)  # Add easting
    northing = models.FloatField(null=True, blank=True) # Add northing

    class Meta:
        indexes = [
            models.Index(fields=['project']),
            models.Index(fields=['prospect']),
        ]

    def __str__(self):
        return f"Trou de forage {self.hole_id}"


class DrillInterval(models.Model):
    """Table des intervalles de mesure avec géologie"""
    id = models.AutoField(primary_key=True)
    drill_hole = models.ForeignKey(DrillHole, on_delete=models.CASCADE, related_name='intervals')
    depth_from = models.FloatField()
    depth_to = models.FloatField()
    lithology = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['drill_hole', 'depth_from', 'depth_to']),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(depth_to__gt=models.F('depth_from')),
                name='depth_check'
            )
        ]

    def __str__(self):
        return f"Intervalle {self.depth_from}-{self.depth_to} du trou {self.drill_hole.hole_id}"

    @property
    def interval_id(self):
        """Génère un ID unique pour l'intervalle"""
        return f"{self.drill_hole.hole_id}{self.depth_from}{self.depth_to}"


class ChemicalAnalysis(models.Model):
    """Table des analyses chimiques"""
    id = models.AutoField(primary_key=True)
    interval = models.ForeignKey(DrillInterval, on_delete=models.CASCADE, related_name='chemical_analyses')

    class Meta:
        indexes = [
            models.Index(fields=['interval']),
        ]

    def __str__(self):
        return f"Analyse chimique de l'intervalle {self.interval}"


class ElementValue(models.Model):
    """Table pour stocker les valeurs des éléments chimiques"""
    id = models.AutoField(primary_key=True)
    analysis = models.ForeignKey(ChemicalAnalysis, on_delete=models.CASCADE, related_name='element_values')
    element = models.CharField(max_length=20)  # Augmenté pour accommoder les noms plus longs
    value = models.FloatField()
    unit = models.CharField(max_length=10)  # ppm, ppb, pct

    class Meta:
        indexes = [
            models.Index(fields=['element']),
            models.Index(fields=['analysis']),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(value__gte=0),
                name='non_negative_value'
            )
        ]

    def __str__(self):
        return f"{self.element} ({self.value} {self.unit})"