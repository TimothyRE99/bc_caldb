"""Reader dataclasses to ingest BlackCAT CalDB keywords and files and
expose them for use in python.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt

from bc_caldb.caldb_generators import (
    GenerateBadpix,
    GenerateCalDB,
    GenerateCodedMask,
    GenerateTeldef,
)
from bc_caldb.constants import CURRENT_CALDB_VER, LIST_KEYWORD_SEP, LIST_KEYWORDS


@dataclass(frozen=True)
class CalDB(ABC):
    """Abstract class for shared methods all BlackCAT CalDB
    readers use.
    """

    @classmethod
    def from_caldb_file(cls, caldb_file: PathLike | str):
        """Generate a CalDB dataclass by reading in a CalDB fits file.

        Not currently implemented.
        """
        # TODO: Implement converting of list keywords back to lists
        # keyword = keyword.split(LIST_KEYWORD_SEP)
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_caldb_version(
        cls, generator: GenerateCalDB, caldb_version: str = CURRENT_CALDB_VER
    ):
        """Use a BlackCAT CalDB generator to generate a CalDB
        dataclass.

        Arguments:
            generator: Which CalDB generator class to use
            caldb_version: BlackCAT CalDB version string. Defaults to
            the most recent version.
        """
        caldb_generator = generator(caldb_version)
        return cls(**caldb_generator.generation_keywords)


@dataclass(frozen=True)
class Teldef(CalDB):
    """Reader for BlackCAT Teldef CalDB. Exposes the CalDB keywords to
    be used in python.
    """

    ccls0001: str
    ccnm0001: str
    cdtp0001: str
    cvsd0001: str
    cvst0001: str
    cdes0001: str

    ncoords: int
    coord0: str
    coord1: str
    coord2: str

    det_ids: list[int]
    raw_xsiz: float
    rawxpix1: float
    raw_xscl: float
    raw_xcol: str
    raw_ysiz: float
    rawypix1: float
    raw_yscl: float
    raw_ycol: str
    raw_unit: str

    detx_min: float
    detx_max: float
    det_xcol: str
    dety_min: float
    dety_max: float
    det_ycol: str
    det_unit: str

    d0_x0: float
    d0_y0: float
    d0_dxdcl: float
    d0_dydcl: float
    d0_dxdrw: float
    d0_dydrw: float
    d1_x0: float
    d1_y0: float
    d1_dxdcl: float
    d1_dydcl: float
    d1_dxdrw: float
    d1_dydrw: float
    d2_x0: float
    d2_y0: float
    d2_dxdcl: float
    d2_dydcl: float
    d2_dxdrw: float
    d2_dydrw: float
    d3_x0: float
    d3_y0: float
    d3_dxdcl: float
    d3_dydcl: float
    d3_dxdrw: float
    d3_dydrw: float

    focallen: float

    optaxisx: int
    optaxisy: int

    def rawxy_to_detxy(
        self,
        rawxs: npt.NDArray[np.floating[Any] | np.integer[Any]],
        rawys: npt.NDArray[np.floating[Any] | np.integer[Any]],
        detids: npt.NDArray[np.floating[Any] | np.integer[Any]],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Use provided CalDB values to convert from RAW coordinates to DET coordinates.

        Arguments:
            rawxs: Numpy array of RAWX values
            rawys: Numpy array of RAWY values
            detids: Numpy array of which detector each (RAWX, RAWY)
            pair comes from
        """
        detxs = np.zeros(rawxs.shape, dtype=np.float64)
        detys = np.zeros(rawys.shape, dtype=np.float64)

        for detid in self.det_ids:
            detid_mask = detids == detid
            detxs[detid_mask] = (
                getattr(self, f"d{detid}_x0")
                + rawxs[detid_mask] * getattr(self, f"d{detid}_dxdcl")
                + rawys[detid_mask] * getattr(self, f"d{detid}_dxdrw")
            )
            detys[detid_mask] = (
                getattr(self, f"d{detid}_y0")
                + rawxs[detid_mask] * getattr(self, f"d{detid}_dydcl")
                + rawys[detid_mask] * getattr(self, f"d{detid}_dydrw")
            )

        return detxs, detys

    @classmethod
    def from_caldb_version(cls, caldb_version: str = CURRENT_CALDB_VER):
        """Use a BlackCAT CalDB generator to generate a Teldef
        dataclass.

        Arguments:
            caldb_version: BlackCAT CalDB version string. Defaults to
            the most recent version.
        """
        return super().from_caldb_version(GenerateTeldef, caldb_version)


@dataclass(frozen=True)
class CodedMask(CalDB):
    """Reader for BlackCAT Aperture CalDB. Exposes the CalDB keywords to
    be used in python.
    """

    ccls0001: str
    ccnm0001: str
    cdtp0001: str
    cvsd0001: str
    cvst0001: str
    cdes0001: str

    ctype1: str
    crpix1: float
    crval1: float
    crunit1: str
    cdelt1: float
    ctype2: str
    crpix2: float
    crval2: float
    crunit2: str
    cdelt2: float

    maskdetx: float
    maskdety: float
    maskdetz: float
    maskoffx: float
    maskoffy: float
    maskoffz: float
    maskpsix: float
    maskpsiy: float
    maskpsiz: float

    maskcelx: float
    maskcely: float
    maskcelz: float

    detdetx: float
    detdety: float
    detdetz: float
    detoffx: float
    detoffy: float
    detoffz: float

    detpixx: float
    detpixy: float
    detpixz: float
    detsizex: float
    detsizey: float
    detsizez: float

    mask_pattern: npt.NDArray[np.bool_]
    frame_pattern: npt.NDArray[np.bool_]

    @classmethod
    def from_caldb_version(cls, caldb_version: str = CURRENT_CALDB_VER):
        """Use a BlackCAT CalDB generator to generate an Aperture
        dataclass.

        Arguments:
            caldb_version: BlackCAT CalDB version string. Defaults to
            the most recent version.
        """
        return super().from_caldb_version(GenerateCodedMask, caldb_version)


@dataclass(frozen=True)
class Badpix(CalDB):
    """Reader for BlackCAT Badpix CalDB. Exposes the CalDB keywords to
    be used in python.
    """

    ccls0001: str
    ccnm0001: str
    cdtp0001: str
    cvsd0001: str
    cvst0001: str
    cdes0001: str

    breason: str
    goodval: int

    badpix_0: npt.NDArray[np.bool_]
    badpix_1: npt.NDArray[np.bool_]
    badpix_2: npt.NDArray[np.bool_]
    badpix_3: npt.NDArray[np.bool_]

    @classmethod
    def from_caldb_version(cls, caldb_version: str = CURRENT_CALDB_VER):
        """Use a BlackCAT CalDB generator to generate an Aperture
        dataclass.

        Arguments:
            caldb_version: BlackCAT CalDB version string. Defaults to
            the most recent version.
        """
        return super().from_caldb_version(GenerateBadpix, caldb_version)
