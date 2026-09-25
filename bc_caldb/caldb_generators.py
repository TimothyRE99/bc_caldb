"""Generator classes to produce BlackCAT CalDB keywords and files."""

from abc import ABC, abstractmethod
from datetime import datetime, UTC
from enum import StrEnum
from functools import cached_property
from importlib import resources
from os import PathLike
from pathlib import Path
from typing import Any, Optional

from astropy.io import fits
import numpy as np
import numpy.typing as npt

from bc_caldb.constants import (
    CURRENT_CALDB_VER,
    DEFAULT_MASK_SEED,
    LIST_KEYWORDS,
    LIST_KEYWORD_SEP,
)


MANDATORY_KEYWORDS = {
    "DATE": (
        datetime.now(tz=UTC).strftime(f"%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "Creation Date",
    ),
    "TELESCOP": ("BLACKCAT", "Telescope (mission) name"),
    "INSTRUME": ("BLACKCAT", "Instrument Name"),
}
MANDATORY_TABLE_KEYWORDS = {
    "ORIGIN": ("PENNSTATE", "Source of FITS file"),
    "CREATOR": ("BC_CALDB", "Software that created FITS file"),
}
SEP = "-" * 70


class MaskVersions(StrEnum):
    """Tracker for BlackCAT CalDB Versions in which the Mask changed."""

    DEFAULT = ""


class TeldefVersions(StrEnum):
    """Tracker for BlackCAT CalDB Versions in which the Teldef
    changed.
    """

    DEFAULT = ""
    V20260614 = "20260614"


class BadpixVersions(StrEnum):
    """Tracker for BlackCAT CalDB Versions in which the Badpix maps
    changed.
    """

    DEFAULT = ""
    V20260423 = "20260423"
    V20260430 = "20260430"
    V20260504 = "20260504"
    V20260508 = "20260508"
    V20260616 = "20260616"
    V20260619 = "20260619"
    V20260625 = "20260625"
    V20260701 = "20260701"
    V20260703 = "20260703"
    V20260722 = "20260722"
    V20260728 = "20260728"
    V20260804 = "20260804"
    V20260811 = "20260811"
    V20260819 = "20260819"
    V20260826 = "20260826"
    V20260831 = "20260831"
    V20260902 = "20260902"
    V20260903 = "20260903"
    V20260904 = "20260904"


CALDB_VERSIONS = {
    ver for ctype in [BadpixVersions, MaskVersions, TeldefVersions] for ver in ctype
}


class GenerateCalDB(ABC):
    """Abstract class for shared methods all BlackCAT CalDB
    generators use.
    """

    CONTENT_DESCRIPTION: str
    DATA_TYPE: str
    DET_IDS = [0, 1, 2, 3]

    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        """CalDB keyword generator

        Arguments:
            caldb_version: BlackCAT CalDB version string. Defaults to
            the most recent version.
        """
        if not isinstance(caldb_version, str | None):
            raise TypeError(
                f"expected caldb_version to be str or None, got {type(caldb_version)}"
            )
        self._caldb_version = caldb_version if caldb_version is not None else ""
        if self._caldb_version not in CALDB_VERSIONS:
            raise ValueError(f"invalid caldb_version: {self._caldb_version}")
        self._generate_caldb_values()

    @cached_property
    def generation_keywords(self) -> dict[str, Any]:
        """Dictionary of all keywords used to read this CalDB."""
        generation_keywords = {
            key.lower(): val[0]
            for tuple in self._commented_dicts
            for key, val in tuple[0].items()
        }

        return generation_keywords

    @cached_property
    def outname(self) -> str:
        """File name that this CalDB will be saved as."""
        return f"bl{self.DATA_TYPE}{self._caldb_version}v{self.version:03}.fits.gz"

    @cached_property
    def version(self) -> int:
        """Which version of a given CalDB date is being used. Currently
        there is only support for one version per date.
        """
        # NOTE: Currently no support for versions other than 001.
        return 1

    @abstractmethod
    def _generate_caldb_values(self) -> None:
        # Generates the necessary values for a given CalDB given a
        # specific version.
        self._commented_dicts: list[tuple[dict[str, tuple[Any, str]], list[str]]]

    def generate_fits_file(
        self,
        outdir: Optional[PathLike | str] = None,
    ) -> fits.PrimaryHDU:
        """Writes the generated CalDB values to a fits primary hdu.

        If outdir is provided, it will write this hdu to a fits file in
        the provided directory.

        Arguments:
            outdir: (Optional) Path to the directory to write the CalDB
            fits file to.
        """
        header = fits.Header(
            cards=[
                *[
                    (key, value, comment)
                    for key, (value, comment) in MANDATORY_KEYWORDS.items()
                ],
                *[
                    (key, value, comment)
                    for key, (value, comment) in MANDATORY_TABLE_KEYWORDS.items()
                ],
                ("VERSION", self.version, "Extension version number"),
                ("FILENAME", self.outname, "File name"),
                ("CONTENT", self.CONTENT_DESCRIPTION, "File content"),
            ]
        )
        for section_dict, section_comments in self._commented_dicts:
            for key, (value, comment) in section_dict.items():
                if key in LIST_KEYWORDS:
                    value = LIST_KEYWORD_SEP.join(str(val) for val in value)

                header.set(key, value, comment)

            for comment in section_comments:
                header.add_comment(comment, before=next(iter(section_dict)))

        primary_hdu = fits.PrimaryHDU(data=None, header=header)

        if outdir is not None:
            primary_hdu.writeto(Path(outdir) / self.outname, checksum=True)

        return primary_hdu


# TODO: Update with new onboard fixes for 0-indexed RAWX/RAWY, plus new offsets
class GenerateTeldef(GenerateCalDB):
    """Generator for BlackCAT Teldef CalDB file."""

    CONTENT_DESCRIPTION = "BlackCAT telescope definition file"
    DATA_TYPE = "teldef"
    DET_PITCH_M = 40e-6
    NOMINAL_GAP_M = 1788e-6
    NUM_SUBPIXELS = 3
    RAW_SIZE = 550

    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        self._teldef_version = max(
            [ver for ver in TeldefVersions if ver <= caldb_version]
        )
        super().__init__(caldb_version)

    @cached_property
    def _c(self) -> float:
        # Nominal distance from optical axis to the center of the
        # subpixel row along the edge of the focal plane.
        return self.DET_PITCH_M * (self.RAW_SIZE - 1 / 6) + self.NOMINAL_GAP_M / 2

    @cached_property
    def _det_offsets_dict(self) -> dict[str, dict[str, npt.NDArray[np.float32]]]:
        # Dictionary holding the offsets from nominal positions for
        # each of the four detectors.
        det_offsets_dict = {
            TeldefVersions.DEFAULT: {
                "x": np.array([0, 0, 0, 0], dtype=np.float32),
                "y": np.array([0, 0, 0, 0], dtype=np.float32),
            },
            TeldefVersions.V20260614: {
                "x": np.array([32.9e-6, 3.1e-6, -232.2e-6, 195.9e-6], dtype=np.float32),
                "y": np.array(
                    [189.0e-6, -167.7e-6, 96.2e-6, -117.6e-6], dtype=np.float32
                ),
            },
        }

        return det_offsets_dict

    @cached_property
    def _dx_dcols(self) -> npt.NDArray[np.float32]:
        # dx_dcol values for each detector
        return np.array(
            [
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
                0,
                0,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _dx_drows(self) -> npt.NDArray[np.float32]:
        # dx_drow values for each detector
        return np.array(
            [
                0,
                0,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _dy_dcols(self) -> npt.NDArray[np.float32]:
        # dy_dcol values for each detector
        return np.array(
            [
                0,
                0,
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _dy_drows(self) -> npt.NDArray[np.float32]:
        # dy_drow values for each detector
        return np.array(
            [
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
                0,
                0,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _x0s(self) -> npt.NDArray[np.float32]:
        # DETX value at the center of the subpixel column along the
        # outside edge of the focal plane for each detector.
        base_x0 = np.array([self._c, -self._c, -self._c, self._c], dtype=np.float32)
        return base_x0 + self._det_offsets_dict[self._teldef_version]["x"]

    @cached_property
    def _y0s(self) -> npt.NDArray[np.float32]:
        # DETY value at the center of the subpixel row along the
        # outside edge of the focal plane for each detector.
        base_y0 = np.array([self._c, -self._c, self._c, -self._c], dtype=np.float32)
        return base_y0 + self._det_offsets_dict[self._teldef_version]["y"]

    def _generate_caldb_values(self) -> None:
        # Generates the necessary values for a given CalDB given a
        # specific version.
        self._commented_dicts = [
            (
                {
                    "CCLS0001": ("BCF", "Dataset is Basic Calibration File"),
                    "CCNM0001": ("TELDEF", "Type of calibration data"),
                    "CDTP0001": ("DATA", "Calibration file contains data"),
                    "CVSD0001": (
                        "2026-02-11",
                        "UTC date when calibration should first be used",
                    ),
                    "CVST0001": (
                        "00:00:00",
                        "UTC time when calibration should first be used",
                    ),
                    "CDES0001": ("TELESCOPE DEFINITION FILE", "Description"),
                },
                [SEP, "CALDB Required Keywords"],
            ),
            (
                {
                    "NCOORDS": (2, "Number of coordinates defined in this file"),
                    "COORD0": ("RAW", "1st coordinate system (DETID, RAWX, RAWY)"),
                    "COORD1": ("DET", "2nd coordinate system (DETX, DETY, DETZ)"),
                    "COORD2": ("SAT", "3rd coordinate system (SATX, SATY, SATZ)"),
                },
                [SEP, "Generic Coordinate Keywords"],
            ),
            (
                {
                    "DET_IDS": (self.DET_IDS, "IDs of included detectors."),
                    "RAW_XSIZ": (
                        self.RAW_SIZE * self.NUM_SUBPIXELS,
                        "RAW space x size (1/3 subpixels)",
                    ),
                    "RAWXPIX1": (
                        0.0,
                        "RAW space x 1st subpix number (1/3 subpixel)",
                    ),
                    "RAW_XSCL": (
                        self.DET_PITCH_M / self.NUM_SUBPIXELS,
                        "RAW X scale (m / subpixel)",
                    ),
                    "RAW_XCOL": ("RAWX", "Name of raw X column in event files"),
                    "RAW_YSIZ": (
                        self.RAW_SIZE * self.NUM_SUBPIXELS,
                        "RAW space y size (1/3 subpixels)",
                    ),
                    "RAWYPIX1": (
                        0.0,
                        "RAW space y 1st subpix number (1/3 subpixel)",
                    ),
                    "RAW_YSCL": (
                        self.DET_PITCH_M / self.NUM_SUBPIXELS,
                        "RAW Y scale (m / subpixel)",
                    ),
                    "RAW_YCOL": ("RAWY", "Name of raw Y column in event files"),
                    "RAW_UNIT": ("1/3 subpixel", "physical unit of RAW coordinates"),
                },
                [
                    SEP,
                    "RAW Coordinate Definition",
                    "These are the subpixel coordinates in the telemetry",
                ],
            ),
            (
                {
                    "DETX_MIN": (np.min(self._x0s), "Minimum det x coordinate"),
                    "DETX_MAX": (np.max(self._x0s), "Maximum det x coordinate"),
                    "DET_XCOL": ("DETX", "Name of DET X column in event files"),
                    "DETY_MIN": (np.min(self._y0s), "Minimum det y coordinate"),
                    "DETY_MAX": (np.max(self._y0s), "Maximum det y coordinate"),
                    "DET_YCOL": ("DETY", "Name of DET Y column in event files"),
                    "DET_UNIT": ("m", "physical unit of DET coordinates"),
                },
                [
                    SEP,
                    "DET coordinates definition",
                    "DET coorindates are fixed to the detector, look-down",
                    "DETZ is positive toward this viewpoint",
                ],
            ),
            (
                {
                    "D0_X0": (self._x0s[0], ""),
                    "D0_Y0": (self._y0s[0], ""),
                    "D0_DXDCL": (self._dx_dcols[0], ""),
                    "D0_DYDCL": (self._dy_dcols[0], ""),
                    "D0_DXDRW": (self._dx_drows[0], ""),
                    "D0_DYDRW": (self._dy_drows[0], ""),
                    "D1_X0": (self._x0s[1], ""),
                    "D1_Y0": (self._y0s[1], ""),
                    "D1_DXDCL": (self._dx_dcols[1], ""),
                    "D1_DYDCL": (self._dy_dcols[1], ""),
                    "D1_DXDRW": (self._dx_drows[1], ""),
                    "D1_DYDRW": (self._dy_drows[1], ""),
                    "D2_X0": (self._x0s[2], ""),
                    "D2_Y0": (self._y0s[2], ""),
                    "D2_DXDCL": (self._dx_dcols[2], ""),
                    "D2_DYDCL": (self._dy_dcols[2], ""),
                    "D2_DXDRW": (self._dx_drows[2], ""),
                    "D2_DYDRW": (self._dy_drows[2], ""),
                    "D3_X0": (self._x0s[3], ""),
                    "D3_Y0": (self._y0s[3], ""),
                    "D3_DXDCL": (self._dx_dcols[3], ""),
                    "D3_DYDCL": (self._dy_dcols[3], ""),
                    "D3_DXDRW": (self._dx_drows[3], ""),
                    "D3_DYDRW": (self._dy_drows[3], ""),
                },
                [
                    SEP,
                    "Translation from RAW to DET coordinates:",
                    "DETX = d#_X0 + RAWX*d#_DXDCL + RAWY*d#DXDRW",
                    "DETY = d#_Y0 + RAWY*d#_DYDCL + RAWY*d#DYDRW",
                ],
            ),
            # TODO: Implement SAT coordinates and DET -> SAT transforms
            (
                {"FOCALLEN": (0.1540, "Telescope focal length (m)")},
                [
                    SEP,
                    "The size of a sky pixel depends on the FPA pixel size, focal length,",
                    "and chosen imaging resolution. 40 microns at a resolution of 1",
                    "corresponds to atan(40e-6 / FOCALLEN) radians on the sky.",
                ],
            ),
            (
                {
                    "OPTAXISX": (0.0, "Optical axis x in DET coordinates (m)"),
                    "OPTAXISY": (0.0, "Optical axis y in DET coordinates (m)"),
                },
                [
                    SEP,
                    "DET is centered on the optical axis, not the distribution of",
                    "detectors on the focal plane.",
                ],
            ),
        ]


class GenerateCodedMask(GenerateCalDB):
    """Generator for BlackCAT Aperture CalDB file."""

    CONTENT_DESCRIPTION = "BlackCAT aperture file"
    DATA_TYPE = "aperture"
    MASK_SHAPE = [249, 555]

    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        self._mask_version = max([ver for ver in MaskVersions if ver <= caldb_version])
        super().__init__(caldb_version)

    @cached_property
    def _det_cent_det_xy(self) -> tuple[float, float]:
        # (DETX, DETY) coordinates of the center of the detector plane.
        # (0, 0) is the optical axis, but shifted detectors means that
        # the center of the detector plane isn't necessary aligned.
        generated_teldef = GenerateTeldef(self._caldb_version)
        detx_center = (
            generated_teldef.generation_keywords["detx_max"]
            + generated_teldef.generation_keywords["detx_min"]
        ) / 2
        dety_center = (
            generated_teldef.generation_keywords["dety_max"]
            + generated_teldef.generation_keywords["dety_min"]
        ) / 2
        return detx_center, dety_center

    @cached_property
    def _frame_pattern(self) -> npt.NDArray[np.bool_]:
        # Pattern showing where the extra support structures are for
        # the mask. Important to track since the mask's 50% open ratio
        # only holds outside the frame.
        pattern = np.zeros(shape=self.MASK_SHAPE, dtype=bool)

        # Main ribs
        for xlow, ylow, xblocksize, yblocksize in self._ribs:
            pattern[ylow : ylow + yblocksize, xlow : xlow + xblocksize] = True

        # Manual fillets
        for xrib_l in [-4, self.MASK_SHAPE[0] // 2, self.MASK_SHAPE[0] + 3]:
            for yrib_l in [-4, self.MASK_SHAPE[1] + 3] + list(self._yribx.astype(int)):
                pattern[
                    max(xrib_l - 4, 0) : xrib_l + 5, max(yrib_l - 7, 0) : yrib_l + 8
                ] = True
                pattern[
                    max(xrib_l - 7, 0) : xrib_l + 8, max(yrib_l - 4, 0) : yrib_l + 5
                ] = True
                pattern[
                    max(xrib_l - 5, 0) : xrib_l + 6, max(yrib_l - 5, 0) : yrib_l + 6
                ] = True

        # Screwhole gussets
        for yrib_l in self._yribx.astype(int):
            for y, ysign in [(0, 1), (self.MASK_SHAPE[0] - 1, -1)]:
                for dy, w in enumerate([19, 15, 13, 11, 9]):
                    pattern[y + dy * ysign, yrib_l - w // 2 : yrib_l + w // 2 + 1] = (
                        True
                    )

        return pattern

    @cached_property
    def generation_keywords(self) -> dict[str, Any]:
        generation_keywords = super().generation_keywords
        generation_keywords["mask_pattern"] = self._mask_pattern
        generation_keywords["frame_pattern"] = self._frame_pattern

        return generation_keywords

    @cached_property
    def _mask_pattern(self) -> npt.NDArray[np.bool_]:
        # Coded aperture mask pattern.
        ny, nx = self.MASK_SHAPE

        pattern = (
            np.array(self.shift_reg_seq(nx * ny, DEFAULT_MASK_SEED), dtype=bool)
            .reshape(self.MASK_SHAPE[::-1])
            .T
        )

        return pattern & ~self._frame_pattern

    @cached_property
    def _ribs(self) -> npt.NDArray[np.uint16]:
        # Pattern showing where the support ribs are. Used to construct
        # the frame pattern when combined with fillets and screw
        # gussets.
        ribhw = 3.5
        xriby = self.MASK_SHAPE[0] / 2
        xcellcount = self.MASK_SHAPE[1]
        yribx = self._yribx
        ycellcount = self.MASK_SHAPE[0]

        ribs = np.array(
            [
                (yribx[0] - ribhw, 0, 2 * ribhw, ycellcount),
                (yribx[1] - ribhw, 0, 2 * ribhw, ycellcount),
                (yribx[2] - ribhw, 0, 2 * ribhw, ycellcount),
                (0, xriby - ribhw, xcellcount, 2 * ribhw),
            ],
            dtype=np.uint16,
        )

        return ribs

    @cached_property
    def _yribx(self) -> npt.NDArray[np.float32]:
        # x locations for the centerlines of the vertical ribs.
        return np.array([138.5, 277.5, 416.5], dtype=np.float32)

    def _generate_caldb_values(self) -> None:
        # Generates the necessary values for a given CalDB given a
        # specific version.
        self._commented_dicts = [
            (
                {
                    "CCLS0001": ("BCF", "Dataset is Basic Calibration File"),
                    "CCNM0001": ("CODED_MASK", "Type of calibration data"),
                    "CDTP0001": ("DATA", "Calibration file contains data"),
                    "CVSD0001": (
                        "2026-02-11",
                        "UTC date when calibration should first be used",
                    ),
                    "CVST0001": (
                        "00:00:00",
                        "UTC time when calibration should first be used",
                    ),
                    "CDES0001": (
                        "BlackCAT Coded mask (aperture) pattern",
                        "Description",
                    ),
                },
                [SEP, "CALDB Required Keywords"],
            ),
            (
                {
                    "CTYPE1": ("DETX", "Title of this axis"),
                    "CRPIX1": (-0.5, "Reference is lower left corner of mask"),
                    "CRVAL1": (
                        -self.MASK_SHAPE[1] * 320e-6 / 2,
                        "Value of DETX at reference point",
                    ),
                    "CRUNIT1": ("m", "Units of DETX"),
                    "CDELT1": (320e-6, "Spacing of cells in m"),
                    "CTYPE2": ("DETY", "Title of this axis"),
                    "CRPIX2": (-0.5, "Reference is lower left corner of mask)"),
                    "CRVAL2": (
                        -self.MASK_SHAPE[0] * 320e-6 / 2,
                        "Value of DETY at reference point",
                    ),
                    "CRUNIT2": ("m", "Units of DETY"),
                    "CDELT2": (320e-6, "Spacing of cells in m"),
                },
                [SEP, "BlackCAT aperture header"],
            ),
            (
                {
                    "MASKDETX": (0.0, "[m] Center of mask cell plane in DETX"),
                    "MASKDETY": (0.0, "[m] Center of mask cell plane in DETY"),
                    "MASKDETZ": (0.1540, "[m] Top of mask cell plane in DETZ"),
                    "MASKOFFX": (0.0, "[m] Offset of mask in DETX"),
                    "MASKOFFY": (0.0, "[m] Offset of mask in DETY"),
                    "MASKOFFZ": (0.0, "[m] Offset of mask in DETZ"),
                    "MASKPSIX": (0.0, "[deg] Mask Euler rotation about X-axis"),
                    "MASKPSIY": (0.0, "[deg] Mask Euler rotation about Y-axis"),
                    "MASKPSIZ": (0.0, "[deg] Mask Euler rotation about Z-axis"),
                },
                [SEP, "Mask position and orientation parameters"],
            ),
            (
                {
                    "MASKCELX": (295e-6, "[m] Size of mask cell in DETX"),
                    "MASKCELY": (295e-6, "[m] Size of mask cell in DETY"),
                    "MASKCELZ": (21e-6, "[m] Size of mask cell in DETZ"),
                },
                [SEP, "Mask cell properties"],
            ),
            (
                {
                    "DETDETX": (0.0, "[m] Top of detector plane in DETX"),
                    "DETDETY": (
                        self._det_cent_det_xy[1],
                        "[m] Center of detector plane in DETY",
                    ),
                    "DETDETZ": (
                        self._det_cent_det_xy[0],
                        "[m] Center of detector plane in DETZ",
                    ),
                    "DETOFFX": (0.0, "[m] Offset of detector plane in DETX"),
                    "DETOFFY": (0.0, "[m] Offset of detector plane in DETY"),
                    "DETOFFZ": (0.0, "[m] Offset of detector plane in DETZ"),
                },
                [SEP, "Detector plane position parameters"],
            ),
            (
                {
                    "DETPIXX": (40e-6, "[m] Size of detector pitch pixel in DETX"),
                    "DETPIXY": (40e-6, "[m] Size of detector pitch pixel in DETX"),
                    "DETPIXZ": (100e-6, "[m] Size of detector pitch pixel in DETX"),
                    "DETSIZEX": (40e-6, "[m] Size of detector pixel in DETX"),
                    "DETSIZEY": (40e-6, "[m] Size of detector pixel in DETX"),
                    "DETSIZEZ": (100e-6, "[m] Size of detector pixel in DETX"),
                },
                [SEP, "Detector size properties"],
            ),
        ]

    def generate_fits_file(
        self,
        outdir: Optional[PathLike | str] = None,
    ) -> fits.HDUList:
        """Writes the generated CalDB values and mask/frame patterns
        to a fits HDUList.

        If outdir is provided, it will write this hdu to a fits file in
        the provided directory.

        Arguments:
            outdir: (Optional) Path to the directory to write the CalDB
            fits file to.
        """
        primary_hdu = super().generate_fits_file(outdir=None)

        mask_ext_header = fits.Header(
            cards=[
                ("EXTNAME", "CODED_MASK", "Name of the image extension"),
                *[
                    (key, value, comment)
                    for key, (value, comment) in MANDATORY_KEYWORDS.items()
                ],
                *[
                    (key, value, comment)
                    for key, (value, comment) in MANDATORY_TABLE_KEYWORDS.items()
                ],
                ("VERSION", self.version, "Extension version number"),
                ("FILENAME", self.outname, "File name"),
                ("CONTENT", "BlackCAT coded mask aperture pattern", "File content"),
                *[("COMMENT", comment) for comment in self._commented_dicts[0][1]],
                *[
                    (key, value, comment)
                    for key, (value, comment) in self._commented_dicts[0][0].items()
                ],
            ]
        )
        mask_ext = fits.ImageHDU(
            data=self._mask_pattern.astype(np.uint8), header=mask_ext_header
        )

        frame_ext_header = fits.Header(
            cards=[
                ("EXTNAME", "SUPPORT_FRAME", "Name of the image extension"),
                *[
                    (key, value, comment)
                    for key, (value, comment) in MANDATORY_KEYWORDS.items()
                ],
                *[
                    (key, value, comment)
                    for key, (value, comment) in MANDATORY_TABLE_KEYWORDS.items()
                ],
                (
                    "CONTENT",
                    "BlackCAT coded mask support frame pattern",
                    "File content",
                ),
                ("FILENAME", self.outname, "File name"),
                ("VERSION", self.version, "Extension version number"),
                *[("COMMENT", comment) for comment in self._commented_dicts[0][1]],
                *[
                    (key, value, comment)
                    for key, (value, comment) in self._commented_dicts[0][0].items()
                ],
            ]
        )
        frame_ext = fits.ImageHDU(
            data=self._frame_pattern.astype(np.uint8), header=frame_ext_header
        )

        hdul = fits.HDUList([primary_hdu, mask_ext, frame_ext])

        if outdir is not None:
            hdul.writeto(Path(outdir) / self.outname, checksum=True)

        return hdul

    @staticmethod
    def shift_reg_seq(seql: int, seed: int) -> npt.NDArray[np.bool_]:
        """Given a sequence length and seed, generates a LFSR sequence
        for building a mask pattern.
        """
        nbits = int(np.ceil(np.log2(seql + 1)))

        taplist = [
            -1,
            -1,
            -1,
            2,
            3,
            3,
            5,
            6,
            -1,
            5,
            7,
            9,
            -1,
            -1,
            -1,
            14,
            -1,
            14,
            11,
            -1,
            17,
            19,
            21,
            18,
            -1,
            22,
            -1,
            -1,
            25,
            27,
            -1,
            28,
        ]
        if taplist[nbits] == -1:
            raise RuntimeError(
                f"No single-tap maximal LFSR with {nbits} bits: specify multiple taps"
            )
        taps = [nbits, taplist[nbits]]

        seq = np.zeros(seql, dtype=bool)
        for idx, seedbit in enumerate(reversed(f"{seed:b}")):
            if idx >= nbits:
                break
            seq[idx] = seedbit == "1"
        for idx in range(nbits, seql):
            for tap in taps:
                seq[idx] ^= seq[idx - tap]

        return seq


class GenerateBadpix(GenerateCalDB):
    """Generator for BlackCAT Badpix CalDB file."""

    CONTENT_DESCRIPTION = "BlackCAT badpix file"
    DATA_TYPE = "badpix"
    OGIP_KEYWORDS = {
        "HDUDOC": ("CAL_GEN_2004_001", "Document describing the format"),
        "HDUCLASS": ("OGIP", "Conforms to OGIP/GSFC standards"),
        "HDUCLAS1": ("IMAGE", "Contains array data"),
        "HDUCLAS2": ("DETMAP", "Histogram is unweighted"),
    }
    # TODO: How should we set these up to properly indicate the scaling being in pixel space, not RAWX/RAWY space?
    REFERENCE_KEYWORDS = {}

    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        self._badpix_version = max(
            [ver for ver in BadpixVersions if ver <= caldb_version]
        )
        super().__init__(caldb_version)

    @cached_property
    def _badpix_date_times(self) -> tuple[str, str]:
        if self._badpix_path is None:
            return "2026-02-11", "00:00:00"

        header = fits.getheader(self._badpix_path)
        yyyy = self._badpix_version[:4]
        mm = self._badpix_version[4:6]
        dd = self._badpix_version[6:8]
        return f"{yyyy}-{mm}-{dd}", header["CVST0001"]

    @cached_property
    def _badpix_path(self) -> Optional[Path]:
        if self._badpix_version == "":
            return None
        else:
            return Path(
                resources.files("bc_caldb.data.badpix_maps").joinpath(
                    f"badpix_{self._badpix_version}.fits.gz"
                )
            )

    @cached_property
    def _badpix_patterns(
        self,
    ) -> tuple[
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
    ]:
        if self._badpix_path is None:
            return (
                np.zeros((550, 550), dtype=bool),
                np.zeros((550, 550), dtype=bool),
                np.zeros((550, 550), dtype=bool),
                np.zeros((550, 550), dtype=bool),
            )

        with fits.open(self._badpix_path) as hdul:
            return (
                hdul[("badpix", 0)].data.astype(bool),
                hdul[("badpix", 1)].data.astype(bool),
                hdul[("badpix", 2)].data.astype(bool),
                hdul[("badpix", 3)].data.astype(bool),
            )

    @cached_property
    def _badpix_reasons(self) -> str:
        # TODO: Fill out reasons
        return {
            BadpixVersions.DEFAULT: "",
            BadpixVersions.V20260423: "",
            BadpixVersions.V20260430: "",
            BadpixVersions.V20260504: "",
            BadpixVersions.V20260508: "",
            BadpixVersions.V20260616: "",
            BadpixVersions.V20260619: "",
            BadpixVersions.V20260625: "",
            BadpixVersions.V20260701: "",
            BadpixVersions.V20260703: "",
            BadpixVersions.V20260722: "",
            BadpixVersions.V20260728: "",
            BadpixVersions.V20260804: "",
            BadpixVersions.V20260811: "",
            BadpixVersions.V20260819: "",
            BadpixVersions.V20260826: "",
            BadpixVersions.V20260831: "",
            BadpixVersions.V20260902: "",
            BadpixVersions.V20260903: "",
            BadpixVersions.V20260904: "",
        }

    @cached_property
    def generation_keywords(self) -> dict[str, Any]:
        generation_keywords = super().generation_keywords
        generation_keywords["badpix_0"] = self._badpix_patterns[0]
        generation_keywords["badpix_1"] = self._badpix_patterns[1]
        generation_keywords["badpix_2"] = self._badpix_patterns[2]
        generation_keywords["badpix_3"] = self._badpix_patterns[3]

        return generation_keywords

    def _generate_caldb_values(self) -> None:
        # Generates the necessary values for a given CalDB given a
        # specific version.
        self._commented_dicts = [
            (
                {
                    "CCLS0001": ("BCF", "Dataset is Basic Calibration File"),
                    "CCNM0001": ("BADPIX", "Type of calibration data"),
                    "CDTP0001": ("DATA", "Calibration file contains data"),
                    "CVSD0001": (
                        self._badpix_date_times[0],
                        "UTC date when calibration should first be used",
                    ),
                    "CVST0001": (
                        self._badpix_date_times[1],
                        "UTC time when calibration should first be used",
                    ),
                    "CDES0001": (
                        "BlackCAT global quality map",
                        "Description",
                    ),
                },
                [],
            ),
            (
                {
                    "BREASON": (
                        self._badpix_reasons[self._badpix_version],
                        "Reason for map transition",
                    ),
                    "GOODVAL": (0, "Good pixels have a map value of zero"),
                },
                [],
            ),
        ]

    def generate_fits_file(
        self,
        outdir: Optional[PathLike | str] = None,
    ) -> fits.HDUList:
        """Writes the generated CalDB values and badpix patterns
        to a fits HDUList.

        If outdir is provided, it will write this hdu to a fits file in
        the provided directory.

        Arguments:
            outdir: (Optional) Path to the directory to write the CalDB
            fits file to.
        """
        primary_hdu = super().generate_fits_file(outdir=None)

        hdu_list = [primary_hdu]
        for detid in self.DET_IDS:
            badpix_ext_header = fits.Header(
                cards=[
                    ("EXTNAME", f"BADPIX_{detid}", "Name of the image extension"),
                    *[
                        (key, value, comment)
                        for key, (value, comment) in self.OGIP_KEYWORDS.items()
                    ],
                    *[
                        (key, value, comment)
                        for key, (value, comment) in MANDATORY_KEYWORDS.items()
                    ],
                    *[
                        (key, value, comment)
                        for key, (value, comment) in MANDATORY_TABLE_KEYWORDS.items()
                    ],
                    ("VERSION", self.version, "Extension version number"),
                    ("FILENAME", self.outname, "File name"),
                    (
                        "CONTENT",
                        f"BlackCAT badpix pattern for detector {detid}",
                        "File content",
                    ),
                    *[("COMMENT", comment) for comment in self._commented_dicts[0][1]],
                    *[
                        (key, value, comment)
                        for key, (value, comment) in self._commented_dicts[0][0].items()
                    ],
                    *[
                        (key, value, comment)
                        for key, (value, comment) in self.REFERENCE_KEYWORDS.items()
                    ],
                    *[("COMMENT", comment) for comment in self._commented_dicts[1][1]],
                    *[
                        (key, value, comment)
                        for key, (value, comment) in self._commented_dicts[1][0].items()
                    ],
                ]
            )
            badpix_ext = fits.ImageHDU(
                data=self._badpix_patterns[detid].astype(np.uint8),
                header=badpix_ext_header,
            )
            hdu_list.append(badpix_ext)

        hdul = fits.HDUList(hdu_list)

        if outdir is not None:
            hdul.writeto(Path(outdir) / self.outname, checksum=True)

        return hdul
