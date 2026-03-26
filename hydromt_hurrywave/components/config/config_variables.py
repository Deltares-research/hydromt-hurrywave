from datetime import datetime, timedelta
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class HurrywaveConfigVariables(BaseSettings):
    """Pydantic model for HurryWave configuration variables (hurrywave.inp).

    Fields follow the order and defaults of the HurryWave kernel's
    ``read_hurrywave_input`` subroutine in ``hurrywave_input.f90``.

    Write policy (controls which fields appear in hurrywave.inp):

    * ``json_schema_extra={"always": True}`` — always written, even when the
      value equals the field default.  Use for fields that the Fortran kernel
      always needs (time control, CRS).
    * No ``json_schema_extra`` (default) — written only when the value differs
      from the field default.  Use for optional physics/numerical parameters.
    * ``json_schema_extra={"condition": "<expr>"}`` — written only when the
      Python expression *<expr>* evaluates to ``True`` against the current
      model-field values **and** the value differs from the field default.
      Use for parameters that are only meaningful under certain settings
      (e.g. ST6-only coefficients).

    Boolean fields (Fortran ``logical``) are stored as Python ``bool``; the
    config writer serialises them to ``0``/``1`` which the Fortran reader
    understands.
    """

    class Config:
        extra = "allow"  # allow unknown parameters to pass through

    # ---- Grid ----
    mmax: Optional[int] = Field(None, ge=1, description="Number of grid cells in m-direction (regular grid)")
    nmax: Optional[int] = Field(None, ge=1, description="Number of grid cells in n-direction (regular grid)")
    dx: Optional[float] = Field(None, gt=0, description="Grid cell size in x-direction")
    dy: Optional[float] = Field(None, gt=0, description="Grid cell size in y-direction")
    x0: Optional[float] = Field(None, description="Grid origin x-coordinate")
    y0: Optional[float] = Field(None, description="Grid origin y-coordinate")
    dt: float = Field(120.0, gt=0, description="Computational time step (seconds)")
    rotation: Optional[float] = Field(None, description="Grid rotation (degrees, anti-clockwise from east)")

    # ---- Time (always written) ----
    tref: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Reference time for the simulation",
        json_schema_extra={"always": True},
    )
    tstart: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Simulation start time",
        json_schema_extra={"always": True},
    )
    tstop: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2),
        description="Simulation stop time",
        json_schema_extra={"always": True},
    )
    tspinup: float = Field(0.0, ge=0, description="Spin-up duration added to tstart (seconds)")
    t0out: float = Field(-999.0, description="Output start time offset from tref (seconds; -999 = use tstart)")
    t1out: float = Field(-999.0, description="Output stop time offset from tref (seconds; -999 = use tstop)")

    # ---- Output intervals ----
    dtmapout: float = Field(0.0, ge=0, description="Map output interval (seconds); keyword 'dtout' also accepted")
    dthisout: float = Field(600.0, ge=0, description="Time-series (his) output interval (seconds)")
    dtsp2out: float = Field(3600.0, ge=0, description="Spectral output interval (seconds)")
    dtwnd: float = Field(1800.0, ge=0, description="Wind forcing update interval (seconds)")
    dtrstout: float = Field(0.0, ge=0, description="Restart file output interval (seconds; 0 = no restart output)")
    dtmaxout: float = Field(0.0, ge=0, description="Maximum wave-height output interval (seconds; 0 = disabled)")
    trstout: float = Field(-999.0, description="Single restart output time offset from tref (seconds; -999 = disabled)")

    # ---- Physical constants ----
    rhoa: float = Field(1.25, gt=0, description="Air density (kg/m³)")
    rhow: float = Field(1024.0, gt=0, description="Water density (kg/m³)")

    # ---- I/O format ----
    inputformat: str = Field("asc", description="Input file format ('asc' or 'bin')")
    outputformat: str = Field("asc", description="Output file format ('asc', 'bin', or 'net')")

    # ---- Numerical diffusion ----
    dmx1: float = Field(0.2, description="Implicit diffusion coefficient")
    dmx2: float = Field(1.0e-5, description="Explicit diffusion coefficient")

    # ---- Wave spectrum discretisation ----
    quadruplets: bool = Field(False, description="Include quadruplet nonlinear interactions")
    freqmin: float = Field(0.04, gt=0, description="Minimum frequency (Hz)")
    freqmax: float = Field(0.50, gt=0, description="Maximum frequency (Hz)")
    nsigma: int = Field(24, ge=1, description="Number of frequency bins")
    ntheta: int = Field(36, ge=1, description="Number of directional bins")
    gammajsp: float = Field(3.3, gt=0, description="JONSWAP peak enhancement factor for boundary conditions")

    # ---- CRS (always written) ----
    crs_epsg: int = Field(4326, description="EPSG code", json_schema_extra={"always": True})
    crs_name: str = Field("WGS 84", description="CRS name string")
    crs_type: str = Field("geographic", description="CRS type: 'geographic' or 'projected'")
    crs_utmzone: Optional[str] = Field(None, description="UTM zone string (e.g. '31N'); overrides crs_name/crs_type")

    # ---- Numerical options ----
    spinup_meteo: bool = Field(False, description="Apply meteorological forcing during spin-up period")
    cglim: float = Field(1.0, description="Group velocity limiter (fraction of deep-water value)")
    #iterperc: float = Field(-999.0, description="Iteration convergence threshold (% cells; -999 = no iteration)")
    #iterthresh: float = Field(0.1, description="Iteration residual threshold")
    spwmergefrac: float = Field(0.5, description="Spiderweb wind merging fraction")
    gambr: float = Field(0.73, description="Breaker parameter (gamma) for depth-induced breaking")
    fbed: float = Field(0.019, description="Bottom friction coefficient")
    redopt: int = Field(1, ge=1, description="Refraction/propagation discretisation option")
    cdcap: float = Field(0.0025, gt=0, description="Maximum wind drag coefficient")
    cdfac: float = Field(1.0, description="Wind drag scaling factor")
    winddrag: str = Field("zijlema", description="Wind drag formulation: 'zijlema', 'wu', or 'hwang'")
    refraction: bool = Field(True, description="Enable wave refraction")
    gccorr: bool = Field(True, description="Enable great-circle correction for geographic grids")
    alpha: float = Field(1.0, description="Courant number scaling factor for propagation")
    vmax_zijlema: float = Field(50.0, description="Maximum wind speed for Zijlema drag (m/s)")

    # ---- Output switches ----
    store_wave_age: bool = Field(False, description="Store wave age in map output")
    store_uorb: bool = Field(False, description="Store orbital velocity in map output")
    store_regular_map: bool = Field(False, description="Store quadtree map output on regular grid")

    # ---- Advanced numerical options ----
    use_lfactor: bool = Field(False, description="Use L-factor correction")
    explicit: bool = Field(False, description="Use explicit (rather than semi-implicit) update scheme")
    physics: str = Field("st3", description="Wind physics package: 'st3' or 'st6'")
    #snl_semi_implicit: bool = Field(False, description="Use semi-implicit treatment of SNL interactions")
    swell_dissipation: str = Field("babanin2011", description="Swell dissipation formulation: 'babanin2011' or 'ard2009'")
    aopp: float = Field(0.09, description="Coefficient for opposing-swell dissipation")
    propagation_scheme: int = Field(2, description="Propagation scheme order (1=upwind, 2=QUICK-type)")
    #quick: bool = Field(True, description="Use QUICK scheme for propagation")
    cdia: float = Field(30000000.0, description="Diagonal diffusion coefficient (m²/s)")
    latmaxlim: float = Field(70.0, description="Maximum latitude for great-circle correction (degrees)")

    # ---- ST6-only coefficients (written only when physics == 'st6') ----
    sds6a1: float = Field(
        4.75e-6,
        description="ST6 swell dissipation coefficient a1",
        json_schema_extra={"condition": "physics == 'st6'"},
    )
    sds6a2: float = Field(
        7.00e-5,
        description="ST6 swell dissipation coefficient a2",
        json_schema_extra={"condition": "physics == 'st6'"},
    )
    sds6p1: int = Field(
        4,
        description="ST6 swell dissipation exponent p1",
        json_schema_extra={"condition": "physics == 'st6'"},
    )
    sds6p2: int = Field(
        4,
        description="ST6 swell dissipation exponent p2",
        json_schema_extra={"condition": "physics == 'st6'"},
    )
    feswell: float = Field(
        0.0041,
        description="Swell friction coefficient",
        json_schema_extra={"condition": "physics == 'st6'"},
    )

    # ---- Debugging ----
    profile_mode: bool = Field(False, description="Enable profile mode (set cgy to 0.0)")

    # ---- Domain files ----
    qtrfile: Optional[str] = Field(None, description="Quadtree grid file (.nc)")
    #depfile: Optional[str] = Field(None, description="Bathymetry/depth file")
    #mskfile: Optional[str] = Field(None, description="Mask file")
    #indexfile: Optional[str] = Field(None, description="Index file")

    # ---- Forcing files ----
    bspfile: Optional[str] = Field(None, description="Boundary spectra file (.bsp)")
    bndfile: Optional[str] = Field(None, description="Boundary points file")
    bhsfile: Optional[str] = Field(None, description="Boundary Hs time series file")
    btpfile: Optional[str] = Field(None, description="Boundary Tp time series file")
    bwdfile: Optional[str] = Field(None, description="Boundary wave direction time series file")
    bdsfile: Optional[str] = Field(None, description="Boundary directional spreading time series file")
    bncfile: Optional[str] = Field(None, description="Boundary conditions NetCDF file")
    spwfile: Optional[str] = Field(None, description="Spiderweb (tropical cyclone) wind file")
    wndfile: Optional[str] = Field(None, description="Uniform wind time-series file")
    amufile: Optional[str] = Field(None, description="Gridded wind u-component file (Delft3D AMU format)")
    amvfile: Optional[str] = Field(None, description="Gridded wind v-component file (Delft3D AMV format)")
    wblfile: Optional[str] = Field(None, description="Wave blocking coefficients file (NetCDF)")
    netamuamvfile: Optional[str] = Field(None, description="Gridded wind netCDF file (u and v combined)")

    # ---- Output files ----
    obsfile: Optional[str] = Field(None, description="Regular (bulk-parameter) observation points file")
    ospfile: Optional[str] = Field(None, description="Spectral observation points file")
    writeruntime: int = Field(0, ge=0, description="Write run-time diagnostics to file (1=yes, 0=no)")
    rstfile: Optional[str] = Field(None, description="Restart file")
    debug: bool = Field(False, description="Enable debug output")

    @field_validator("tref", "tstart", "tstop", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y%m%d %H%M%S")
        return v
