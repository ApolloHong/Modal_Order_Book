"""Wrapper de compatibilité pour l'ancien module Hawkes en majuscules.

Le nouveau code devrait importer depuis ``model.hawkes``. Ce fichier est conservé
pour que les notebooks existants utilisant ``from model.Hawkes import ...`` continuent de fonctionner sans modification.
"""

from .hawkes import *  # noqa: F401,F403
