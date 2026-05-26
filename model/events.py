"""
Types d'événements pour le modèle Queue Reactive.

Événements pouvant survenir sur chaque file d'attente :
    ADD     (A)  : Un nouvel ordre limite arrive → taille de la file +1
    CANCEL  (C)  : Un ordre existant est annulé → taille de la file -1
    TRADE   (T)  : Un ordre au marché consomme une unité → taille de la file -1
                   (uniquement sur les premières limites Q_{±1})

Événements spéciaux lorsqu'une première limite est vide :
    BID     (B)  : Un ordre d'achat remplit la limite vide côté vendeur (ask)
                   → déclenche un décalage de prix vers la GAUCHE (baisse du prix)
    ASK     (A)  : Un ordre de vente remplit la limite vide côté acheteur (bid)
                   → déclenche un décalage de prix vers la DROITE (hausse du prix)

Événement étendu (pour les modèles plus complexes) :
    TOTAL_CONSUMPTION (TC) : Un ordre au marché épuise entièrement la première limite
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class EventType(Enum):
    ADD = auto()       # Insertion d'un ordre limite : +1
    CANCEL = auto()    # Annulation d'un ordre : -1
    TRADE = auto()     # Ordre au marché (premières limites uniquement) : -1
    BID = auto()       # Remplissage d'une limite vide depuis le côté achat → décalage de prix
    ASK = auto()       # Remplissage d'une limite vide depuis le côté vente → décalage de prix
    TOTAL_CONSUMPTION = auto()  # Épuisement total de la première limite (extension)


@dataclass
class Event:
    """
    Un événement unique dans la simulation du carnet d'ordres (LOB).
    
    Attributs
    ----------
    time : float
        Temps absolu auquel l'événement se produit.
    event_type : EventType
        Type de l'événement (ADD, CANCEL, TRADE, BID, ASK).
    queue_index : int
        Indice de la file d'attente affectée (par ex. -1 pour le meilleur achat, 1 pour la meilleure vente).
    size : float
        Variation de la taille (généralement +1 ou -1 ; peut différer pour les extensions).
    """
    time: float
    event_type: EventType
    queue_index: int
    size: float = 1.0

    @property
    def is_price_changing(self) -> bool:
        """Cet événement déclenche-t-il un changement du prix de référence ?"""
        return self.event_type in (EventType.BID, EventType.ASK)

    def __repr__(self):
        return (f"Event(t={self.time:.6f}, {self.event_type.name}, "
                f"Q[{self.queue_index}], Δ={self.size:+.0f})")
