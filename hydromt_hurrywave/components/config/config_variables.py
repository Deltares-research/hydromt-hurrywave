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

    Each field also carries a ``"section"`` key in ``json_schema_extra`` that
    groups variables into named blocks in the written hurrywave.inp.  The
    canonical section order is: Time, Domain, Physics, Numerics, Boundaries,
    Meteo, Output, Debug.

    Boolean fields (Fortran ``logical``) are stored as Python ``bool``; the
    config writer serialises them to ``0``/``1`` which the Fortran reader
    understands.
    """

    class Config:
        extra = "allow"  # allow unknown parameters to pass through

    # ---- Time step ----
    dt: float = Field(
        120.0,
        gt=0,
        description="Computational time step (seconds)",
        json_schema_extra={"always": True, "section": "Time"},
    )

    # ---- Time (always written) ----
    tref: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Reference time for the simulation",
        json_schema_extra={"always": True, "section": "Time"},
    )
    tstart: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Simulation start time",
        json_schema_extra={"always": True, "section": "Time"},
    )
    tstop: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2),
        description="Simulation stop time",
        json_schema_extra={"always": True, "section": "Time"},
    )
    tspinup: float = Field(0.0, ge=0, description="Spin-up duration added to tstart (seconds)", json_schema_extra={"always": True, "section": "Time"})
    t0out: float = Field(-999.0, description="Output start time offset from tref (seconds; -999 = use tstart)", json_schema_extra={"section": "Output"})
    t1out: float = Field(-999.0, description="Output stop time offset from tref (seconds; -999 = use tstop)", json_schema_extra={"section": "Output"})

    # ---- Output intervals ----
    dtmapout: float = Field(3600.0, ge=0, description="Map output interval (seconds); keyword 'dtout' also accepted", json_schema_extra={"always": True, "section": "Output"})
    dthisout: float = Field(600.0, ge=0, description="Time-series (his) output interval (seconds)", json_schema_extra={"always": True, "section": "Output"})
    dtsp2out: float = Field(3600.0, ge=0, description="Spectral output interval (seconds)", json_schema_extra={"always": True, "section": "Output"})
    dtwnd: float = Field(1800.0, ge=0, description="Wind forcing update interval (seconds)", json_schema_extra={"always": True, "section": "Meteo"})
    dtrstout: float = Field(0.0, ge=0, description="Restart file output interval (seconds; 0 = no restart output)", json_schema_extra={"section": "Output"})
    dtmaxout: float = Field(0.0, ge=0, description="Maximum wave-height output interval (seconds; 0 = disabled)", json_schema_extra={"always": True, "section": "Output"})
    trstout: float = Field(-999.0, description="Single restart output time offset from tref (seconds; -999 = disabled)", json_schema_extra={"section": "Output"})

    # ---- Physical constants ----
    rhoa: float = Field(1.25, gt=0, description="Air density (kg/m³)", json_schema_extra={"always": True, "section": "Physics"})
    rhow: float = Field(1024.0, gt=0, description="Water density (kg/m³)", json_schema_extra={"always": True, "section": "Physics"})

    # ---- Numerical diffusion ----
    dmx1: float = Field(0.2, description="Implicit diffusion coefficient", json_schema_extra={"section": "Numerics"})

    # ---- Wave spectrum discretisation ----
    quadruplets: bool = Field(False, description="Include quadruplet nonlinear interactions", json_schema_extra={"always": True, "section": "Physics"})
    freqmin: float = Field(0.04, gt=0, description="Minimum frequency (Hz)", json_schema_extra={"always": True, "section": "Domain"})
    freqmax: float = Field(0.50, gt=0, description="Maximum frequency (Hz)", json_schema_extra={"always": True, "section": "Domain"})
    nsigma: int = Field(24, ge=1, description="Number of frequency bins", json_schema_extra={"always": True, "section": "Domain"})
    ntheta: int = Field(36, ge=1, description="Number of directional bins", json_schema_extra={"always": True, "section": "Domain"})
    gammajsp: float = Field(3.3, gt=0, description="JONSWAP peak enhancement factor for boundary conditions", json_schema_extra={"section": "Boundaries"})

    # ---- CRS (always written) ----
    crs_epsg: int = Field(4326, description="EPSG code", json_schema_extra={"always": True, "section": "Domain"})
    crs_name: str = Field("WGS 84", description="CRS name string", json_schema_extra={"section": "Domain"})
    crs_type: str = Field("geographic", description="CRS type: 'geographic' or 'projected'", json_schema_extra={"section": "Domain"})
    crs_utmzone: Optional[str] = Field(None, description="UTM zone string (e.g. '31N'); overrides crs_name/crs_type", json_schema_extra={"section": "Domain"})

    # ---- Numerical options ----
    spinup_meteo: bool = Field(False, description="Apply meteorological forcing during spin-up period", json_schema_extra={"section": "Meteo"})
    spwmergefrac: float = Field(0.5, description="Spiderweb wind merging fraction", json_schema_extra={"section": "Meteo"})
    gambr: float = Field(0.73, description="Breaker parameter (gamma) for depth-induced breaking", json_schema_extra={"section": "Physics"})
    fbed: float = Field(0.019, description="Bottom friction coefficient", json_schema_extra={"section": "Physics"})
    cdcap: float = Field(0.0025, gt=0, description="Maximum wind drag coefficient", json_schema_extra={"always": True, "section": "Meteo"})
    cdfac: float = Field(1.0, description="Wind drag scaling factor", json_schema_extra={"always": True, "section": "Meteo"})
    winddrag: str = Field("zijlema", description="Wind drag formulation: 'zijlema', 'wu', or 'hwang'", json_schema_extra={"always": True, "section": "Meteo"})
    refraction: bool = Field(True, description="Enable wave refraction", json_schema_extra={"always": True, "section": "Physics"})
    gccorr: bool = Field(True, description="Enable great-circle correction for geographic grids", json_schema_extra={"section": "Numerics"})
    alpha: float = Field(1.0, description="Courant number scaling factor for propagation", json_schema_extra={"section": "Numerics"})
    vmax_zijlema: float = Field(50.0, description="Maximum wind speed for Zijlema drag (m/s)", json_schema_extra={"section": "Meteo"})

    # ---- Output switches ----
    store_wave_age: bool = Field(False, description="Store wave age in map output", json_schema_extra={"section": "Output"})
    store_uorb: bool = Field(False, description="Store orbital velocity in map output", json_schema_extra={"section": "Output"})
    store_regular_map: bool = Field(False, description="Store quadtree map output on regular grid", json_schema_extra={"section": "Output"})

    # ---- Advanced numerical options ----
    use_lfactor: bool = Field(False, description="Use L-factor correction", json_schema_extra={"section": "Numerics"})
    explicit: bool = Field(False, description="Use explicit (rather than semi-implicit) update scheme", json_schema_extra={"section": "Numerics"})
    physics: str = Field("st3", description="Wind physics package: 'st3', 'st4' or 'st6'", json_schema_extra={"always": True, "section": "Physics"})
    #snl_semi_implicit: bool = Field(False, description="Use semi-implicit treatment of SNL interactions")
    swell_dissipation: str = Field("babanin2011", description="Swell dissipation formulation: 'babanin2011' or 'ard2009'", json_schema_extra={"section": "Physics"})
    aopp: float = Field(0.09, description="Coefficient for opposing-swell dissipation", json_schema_extra={"section": "Physics"})
    propagation_scheme: int = Field(2, description="Propagation scheme order (1=upwind, 2=QUICK-type)", json_schema_extra={"section": "Numerics"})
    #quick: bool = Field(True, description="Use QUICK scheme for propagation")
    cdia: float = Field(30000000.0, description="Diagonal diffusion coefficient (m²/s)", json_schema_extra={"section": "Numerics"})
    latmaxlim: float = Field(70.0, description="Maximum latitude for great-circle correction (degrees)", json_schema_extra={"section": "Numerics"})

    # ---- ST6-only coefficients (written only when physics == 'st6') ----
    sds6a1: float = Field(
        4.75e-6,
        description="ST6 swell dissipation coefficient a1",
        json_schema_extra={"condition": "physics == 'st6'", "section": "Physics"},
    )
    sds6a2: float = Field(
        7.00e-5,
        description="ST6 swell dissipation coefficient a2",
        json_schema_extra={"condition": "physics == 'st6'", "section": "Physics"},
    )
    sds6p1: int = Field(
        4,
        description="ST6 swell dissipation exponent p1",
        json_schema_extra={"condition": "physics == 'st6'", "section": "Physics"},
    )
    sds6p2: int = Field(
        4,
        description="ST6 swell dissipation exponent p2",
        json_schema_extra={"condition": "physics == 'st6'", "section": "Physics"},
    )
    feswell: float = Field(
        0.0041,
        description="Swell friction coefficient",
        json_schema_extra={"condition": "physics == 'st6'", "section": "Physics"},
    )

    # ---- Debugging ----
    profile_mode: bool = Field(False, description="Enable profile mode (set cgy to 0.0)", json_schema_extra={"section": "Debug"})

    # ---- Domain files ----
    qtrfile: Optional[str] = Field(None, description="Quadtree grid file (.nc)", json_schema_extra={"always": True, "section": "Domain"})
    #depfile: Optional[str] = Field(None, description="Bathymetry/depth file")
    #mskfile: Optional[str] = Field(None, description="Mask file")
    #indexfile: Optional[str] = Field(None, description="Index file")

    # ---- Forcing files ----
    bspfile: Optional[str] = Field(None, description="Boundary spectra file (.bsp)", json_schema_extra={"section": "Boundaries"})
    bndfile: Optional[str] = Field(None, description="Boundary points file", json_schema_extra={"section": "Boundaries"})
    bhsfile: Optional[str] = Field(None, description="Boundary Hs time series file", json_schema_extra={"section": "Boundaries"})
    btpfile: Optional[str] = Field(None, description="Boundary Tp time series file", json_schema_extra={"section": "Boundaries"})
    bwdfile: Optional[str] = Field(None, description="Boundary wave direction time series file", json_schema_extra={"section": "Boundaries"})
    bdsfile: Optional[str] = Field(None, description="Boundary directional spreading time series file", json_schema_extra={"section": "Boundaries"})
    bncfile: Optional[str] = Field(None, description="Boundary conditions NetCDF file", json_schema_extra={"section": "Boundaries"})
    spwfile: Optional[str] = Field(None, description="Spiderweb (tropical cyclone) wind file", json_schema_extra={"section": "Meteo"})
    wndfile: Optional[str] = Field(None, description="Uniform wind time-series file", json_schema_extra={"section": "Meteo"})
    amufile: Optional[str] = Field(None, description="Gridded wind u-component file (Delft3D AMU format)", json_schema_extra={"section": "Meteo"})
    amvfile: Optional[str] = Field(None, description="Gridded wind v-component file (Delft3D AMV format)", json_schema_extra={"section": "Meteo"})
    wblfile: Optional[str] = Field(None, description="Wave blocking coefficients file (NetCDF)", json_schema_extra={"section": "Domain"})
    netamuamvfile: Optional[str] = Field(None, description="Gridded wind netCDF file (u and v combined)", json_schema_extra={"section": "Meteo"})

    # ---- Output files ----
    obsfile: Optional[str] = Field(None, description="Regular (bulk-parameter) observation points file", json_schema_extra={"section": "Output"})
    ospfile: Optional[str] = Field(None, description="Spectral observation points file", json_schema_extra={"section": "Output"})
    writeruntime: int = Field(0, ge=0, description="Write run-time diagnostics to file (1=yes, 0=no)", json_schema_extra={"section": "Output"})
    rstfile: Optional[str] = Field(None, description="Restart file", json_schema_extra={"section": "Output"})
    debug: bool = Field(False, description="Enable debug output", json_schema_extra={"section": "Debug"})

    # ---- Wave forces / extra output switches ----
    wave_force_option: Optional[str] = Field(None, description="Wave force formulation (e.g. 'vor')", json_schema_extra={"section": "Physics"})
    store_wave_forces: Optional[bool] = Field(None, description="Store wave forces in map output", json_schema_extra={"section": "Output"})
    store_wave_vectors: Optional[bool] = Field(None, description="Store wave vector components in map output", json_schema_extra={"section": "Output"})
    output_on_quadtree_mesh: Optional[bool] = Field(None, description="Write map output on the native quadtree mesh", json_schema_extra={"section": "Output"})

    @field_validator("tref", "tstart", "tstop", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y%m%d %H%M%S")
        return v
