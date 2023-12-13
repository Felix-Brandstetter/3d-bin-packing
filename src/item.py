from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import permutations
from operator import itemgetter
from functools import reduce
import copy
import operator
import uuid
#from .names import *
import pickle
from typing import Union, Tuple


class Item():
    """An item that can be placed in a bin."""

    OBS_SIZE = 4  # extents (x,y,z), weight
    N_ROTATIONS = 6

    STPAE = 'STPAE'
    GRPAE = 'GRPAE'

    INVERSE_ROTATIONS = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5}

    def __init__(self, extents: list = [1, 1, 1], weight: int = 1, description: str = '-', packeinheit: str = '-',
                 material_id: str = '', packeinheit_menge: int = None):
        """Create a new Item."""
        assert len(extents) == 3, extents
        assert all([i > 0 for i in extents]), extents
        assert weight > 0
        self.id = uuid.uuid4()
        self.extents = np.array(extents).astype(int)
        # self.extents = np.sort(self.extents)  # sort to have systematic orientation
        self.position = None
        self.weight = weight
        self.description = description
        self.packeinheit = packeinheit
        self.material_id = material_id
        self.packeinheit_menge = packeinheit_menge

    def min(self, dim: int = None) -> int:
        """Returns the minimum coordinate of this item in the given dimension."""
        assert self.position is not None
        minpos = self.position
        if dim is not None:
            assert 0 <= dim < 3
            return minpos[dim]
        return minpos.copy()

    def max(self, dim: int = None) -> int:
        """Returns the maximum coordinate of this item in the given dimension."""
        assert self.position is not None
        maxpos = self.position + self.extents
        if dim is not None:
            assert 0 <= dim < 3
            return maxpos[dim]
        return maxpos

    def mean(self, dim: int = None) -> int:
        """Returns the mean coordinate of this item in the given dimension."""
        assert self.position is not None
        mean = self.position + self.extents / 2
        if dim is not None:
            assert 0 <= dim < 3
            return mean[dim]
        return mean

    def set_position(self, position: list):
        """Sets the position of this item."""
        assert len(position) == 3
        self.position = np.array(position)
        self._set_bottom_corners()
        return self

    def clear_position(self):
        """Clears this item's position."""
        self.position = None
        self.bottom_corners = None

    def has_position(self) -> bool:
        """Returns if this item has a position."""
        return self.position is not None

    def xy_area(self) -> int:
        """Returns the area of this item in xy-dimension."""
        return self.extents[0] * self.extents[1]

    def rotate(self, rotation: int) -> Item:
        """Rotates this item."""
        assert rotation < self.N_ROTATIONS
        perm = list(permutations([0, 1, 2]))[rotation]
        self.extents = np.take(self.extents, perm)
        return self

    def intersects(self, other: Item) -> bool:
        """Returns if this item intersects the given item."""
        assert self.position is not None
        assert other.position is not None
        return not any([Item.is_dimension_separated(self, other, i) for i in np.arange(3)])

    def observation(self) -> np.ndarray:
        """Returns an array containing the observation information of this item."""
        obs = np.concatenate([self.extents, [self.weight]]).astype(int)
        assert len(obs) == Item.OBS_SIZE
        return obs

    def copy(self) -> Item:
        """Creates a deepcopy of this item."""
        return copy.deepcopy(self)  # pickle.loads(pickle.dumps(bin3d))

    def __str__(self) -> str:
        return (f'Item ({self.description}); lwh: {self.extents.tolist()}; pos: {self.position}; '
                f'weight: {self.weight}; volume: {self.volume()}')

    def __repr__(self) -> str:
        return str(self)

    def volume(self) -> int:
        """Returns this item's volume."""
        return reduce(operator.mul, self.extents)

    def weight(self) -> int:
        """Returns this item's weight."""
        return self.weight

    def density(self) -> float:
        """Returns this item's density."""
        return self.weight / self.volume()

    def is_dimension_separated(a: Item, b: Item, i: int) -> bool:
        """Returns if the given dimension i is a separating dimension for the two given items."""
        assert a.has_position() and b.has_position(), f'Calculating separation needs Items with positions ({a}; {b})'
        return (a.min(i) <= b.min(i)) & (a.max(i) <= b.min(i)) | \
               (a.min(i) >= b.max(i)) & (a.max(i) >= b.max(i))

    def support_area(self, other: Item) -> int:
        """Returns the area by how much this item supports the given one.
        (A positive area implies this Item is below the other Item.)"""
        if self.max(2) != other.min(2):  # z value not aligned, so no direct support
            return 0
        intersect_x = interval_intersection([self.min(0), self.max(0)], [other.min(0), other.max(0)])
        intersect_y = interval_intersection([self.min(1), self.max(1)], [other.min(1), other.max(1)])
        if intersect_x > 0 and intersect_y > 0:
            return intersect_x * intersect_y
        else:
            return 0

    def contact_area(self, other: Bin) -> int:
        """Returns the contact surface area of this Item and the given one."""
        a = self
        b = other
        for d0, d1, d2 in ([0, 1, 2], [1, 0, 2], [2, 0, 1]):
            if a.min(d0) == b.max(d0) or a.max(d0) == b.min(d0):
                return contact_area(a, b, d1, d2)
        return 0

    def contact_area_bin(self, other: Bin) -> int:
        """Returns the contact surface area of this Item and the given bin."""
        a = self
        b = other
        area_sum = 0
        for d0, d1, d2 in ([0, 1, 2], [1, 0, 2], [2, 0, 1]):
            # both ends can touch the bin wall, each case seperate
            if a.min(d0) == b.min(d0):
                area_sum += contact_area(a, b, d1, d2)
            if a.max(d0) == b.max(d0):
                area_sum += contact_area(a, b, d1, d2)
        return area_sum

    def create(record: dict, packeinheit: str = None) -> Item:
        """Creates an Item based on a stamm data record."""
        if packeinheit is None:
            packeinheit = Item.STPAE if STPAE_LAENGE in record else Item.GRPAE
        lbh = STPAE_LBH if packeinheit == Item.STPAE else GRPAE_LBH
        weight = STPAE_GEWICHT if packeinheit == Item.STPAE else GRPAE_GEWICHT
        menge = STPAE_MENGE if packeinheit == Item.STPAE else GRPAE_MENGE
        lbh = np.array(itemgetter(*lbh)(record)).astype(float) / 10  # mm to cm
        assert np.allclose(lbh, np.round(lbh)), f'Expected extents in mm rounded to cm. {lbh}'
        return Item(extents=lbh, weight=record[weight], material_id=record[MAT_ID], description=record[BESCHREIBUNG],
                    packeinheit=packeinheit, packeinheit_menge=record[menge])

    def _set_bottom_corners(self):
        """Sets the bottom corners of this Item."""
        mi = self.min()
        ma = self.max()
        p1 = np.array([mi[0], mi[1], mi[2]])
        p2 = np.array([ma[0], mi[1], mi[2]])
        p3 = np.array([ma[0], ma[1], mi[2]])
        p4 = np.array([mi[0], ma[1], mi[2]])
        self.bottom_corners = p1, p2, p3, p4

    def __eq__(self, other: Item) -> bool:
        if other is None:
            return False
        return (
            (self.id == other.id) &
            (self.extents == other.extents).all() &
            np.array(self.position == other.position).all() &
            (self.weight == other.weight) &
            (self.description == other.description) &
            (self.packeinheit == other.packeinheit) &
            (self.material_id == other.material_id) &
            (self.packeinheit_menge == other.packeinheit_menge)
        )

    def __hash__(self):
        return hash((
            self.id,
            tuple(self.extents),
            tuple(self.position) if self.position is not None else None,
            self.weight,
            self.description,
            self.packeinheit,
            self.material_id,
            self.packeinheit_menge,
        ))
    
    def physical_hash(self):
        return hash((
            #tuple(self.position) if self.position is not None else None,
            tuple(self.extents)
        ))    
    

    def equals_physical(self, other: Item) -> bool:
        """Returns this item's physical body is equals to the given one."""
        return (
            (np.sort(self.extents) == np.sort(other.extents)).all() &
            (self.weight == other.weight)
        )


def interval_intersection(a: list, b: list) -> int:
    """Intersection size of two intervals. Can be negative.
    For an explanation see: https://scicomp.stackexchange.com/a/26260 """
    return min(a[1], b[1])-max(a[0], b[0])


def contact_area(a, b, d1: int, d2: int) -> int:
    """Returns the contact area of the two given objects in the plane of the given dimensions."""
    d1_inters = interval_intersection([a.min(d1), a.max(d1)], [b.min(d1), b.max(d1)])
    d1_inters = max(0, d1_inters)  # negative area means no intersection, so set to zero
    d2_inters = interval_intersection([a.min(d2), a.max(d2)], [b.min(d2), b.max(d2)])
    d2_inters = max(0, d2_inters)
    return d1_inters * d2_inters
