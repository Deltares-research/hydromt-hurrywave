import logging
from ast import literal_eval
from datetime import datetime
from os.path import abspath, exists, isabs, join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import ValidationError

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from .config_variables import HurrywaveConfigVariables

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")


class HurrywaveConfig(ModelComponent):
    """Read and write HurryWave configuration files (hurrywave.inp).

    Uses :class:`HurrywaveConfigVariables` (Pydantic) for validation.
    """

    def __init__(self, model: "HurrywaveModel"):
        self._filename = "hurrywave.inp"
        self._data: Optional[HurrywaveConfigVariables] = None
        super().__init__(model=model)

    @property
    def data(self) -> HurrywaveConfigVariables:
        """Return the Pydantic HurrywaveConfigVariables object."""
        if self._data is None:
            self._data = HurrywaveConfigVariables()
            if self.root.is_reading_mode():
                self.read()
        return self._data

    @property
    def filename(self) -> Path:
        """Absolute path to hurrywave.inp."""
        p = Path(self._filename)
        if not p.is_absolute():
            p = self.model.root.path.resolve() / "hurrywave.inp"
        return p

    def read(self) -> None:
        """Read hurrywave.inp and populate HurrywaveConfigVariables."""
        if not exists(self.filename):
            raise FileNotFoundError(f"HurryWave input file '{self.filename}' does not exist.")

        with open(self.filename, "r") as fid:
            lines = fid.readlines()

        inp_dict = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            comment_idx = line.find("#")
            if comment_idx >= 0:
                line = line[:comment_idx]
            parts = [x.strip() for x in line.split("=")]
            if len(parts) != 2:
                continue
            name, val = parts
            if name in ("tref", "tstart", "tstop"):
                try:
                    val = datetime.strptime(val, "%Y%m%d %H%M%S")
                except ValueError:
                    pass
            elif val.lower() in ("yes", "true"):
                val = True
            elif val.lower() in ("no", "false"):
                val = False
            else:
                try:
                    val = literal_eval(val)
                except Exception:
                    pass
            inp_dict[name] = val

        current_data = self.data.model_dump()
        current_data.update(inp_dict)
        try:
            self._data = self.data.__class__.model_validate(current_data)
        except ValidationError as e:
            bad_fields = {err["loc"][0] for err in e.errors()}
            for f in bad_fields:
                logger.warning(
                    f"Skipping invalid config value for '{f}': {current_data.pop(f)!r}"
                )
            self._data = self.data.__class__.model_validate(current_data)

    def write(self, filename: str = "hurrywave.inp", write_description: bool = False) -> None:
        """Write HurrywaveConfigVariables to hurrywave.inp.

        Each field's ``json_schema_extra`` controls whether it is written:

        * ``{"always": True}`` — written unconditionally (e.g. time, CRS).
        * No extra (default) — written only when the value differs from the
          field default.
        * ``{"condition": "<expr>"}`` — written only when *<expr>* evaluates
          to ``True`` against the current field values **and** the value
          differs from the field default.

        Fields are grouped by their ``"section"`` entry in ``json_schema_extra``
        and written under ``# <section>`` headers in the canonical order
        (Time, Domain, Physics, Numerics, Boundaries, Meteo, Output, Debug).
        Any unknown sections are appended at the end.  Extra (undeclared)
        fields land in a trailing ``# Extra`` block.

        ``None`` values are never written regardless of policy.
        Extra fields (not declared in the model) are always written when
        their value is not ``None``.
        """
        if not isabs(filename) and self.root.path:
            self._filename = self.root.path / filename

        self.filename.parent.mkdir(parents=True, exist_ok=True)

        model_fields = HurrywaveConfigVariables.model_fields
        data_dict = self.data.model_dump()

        section_order = [
            "Time",
            "Domain",
            "Physics",
            "Numerics",
            "Boundaries",
            "Meteo",
            "Output",
            "Debug",
        ]
        sections: Dict[str, list] = {}

        for key, value in data_dict.items():
            # Never write None
            if value is None:
                continue

            field_info = model_fields.get(key)
            extra = (field_info.json_schema_extra or {}) if field_info else {}

            # Evaluate condition when present
            condition = extra.get("condition")
            if condition is not None:
                try:
                    if not eval(condition, {}, data_dict):  # noqa: S307
                        continue
                except Exception:
                    pass  # condition evaluation failed — write anyway

            # Decide always vs non-default
            always = extra.get("always", False)
            if not always:
                if field_info is not None:
                    try:
                        if value == field_info.default:
                            continue
                    except Exception:
                        pass  # exotic default type — write anyway
                # Extra fields (not declared) are always written

            # Serialise
            # Booleans -> yes/no (check before numeric branch since bool is an int)
            if isinstance(value, bool):
                line = f"{key.ljust(25)} = {'yes' if value else 'no'}"
            else:
                # Preserve float type if the field is declared as float
                if field_info is not None and field_info.annotation is float:
                    value = float(value) if not isinstance(value, float) else value
                else:
                    value = _convert_to_number(value)

                if isinstance(value, (int, float)):
                    line = f"{key.ljust(25)} = {value}"
                elif isinstance(value, list):
                    line = f"{key.ljust(25)} = {' '.join(str(v) for v in value)}"
                elif hasattr(value, "strftime"):
                    line = f"{key.ljust(25)} = {value.strftime('%Y%m%d %H%M%S')}"
                else:
                    line = f"{key.ljust(25)} = {value}"

            # Add description as comment
            if write_description and field_info and field_info.description:
                line = f"{line.ljust(60)} # {field_info.description}"

            section = extra.get("section") or ("Extra" if field_info is None else "Other")
            sections.setdefault(section, []).append(line)

        ordered = [s for s in section_order if s in sections]
        ordered += [s for s in sections if s not in section_order and s != "Extra"]
        if "Extra" in sections:
            ordered.append("Extra")

        with open(self.filename, "w") as fid:
            for i, section in enumerate(ordered):
                if i > 0:
                    fid.write("\n")
                fid.write(f"# {section}\n")
                for line in sections[section]:
                    fid.write(line + "\n")

    def get(self, key: str, fallback: Any = None, abs_path: bool = False) -> Any:
        """Get a configuration value by key."""
        value = self.data.model_dump().get(key, fallback)
        if value is None and fallback is not None:
            value = fallback
        if abs_path and isinstance(value, (str, Path)):
            value = Path(value)
            if not value.is_absolute():
                value = Path(abspath(join(self.root.path, value)))
        return value

    def set(self, key: str, value: Any, skip_validation: bool = False) -> None:
        """Set a configuration value by key."""
        if not hasattr(self.data, key) and key not in self.data.model_extra:
            logger.warning(f"'{key}' is not a recognised HurrywaveConfig attribute.")
        if not skip_validation:
            new_data = self.data.model_dump()
            new_data[key] = value
            self._data = self.data.__class__.model_validate(new_data)
        else:
            setattr(self._data, key, value)

    @hydromt_step
    def update(
        self,
        dict: Optional[Dict[str, Any]] = None,
        *,
        skip_validation: bool = False,
        **kwargs,
    ) -> None:
        """Update configuration from a dictionary or keyword arguments."""
        updates = dict or {}
        updates.update(kwargs)
        if updates:
            if skip_validation:
                for k, v in updates.items():
                    setattr(self._data, k, v)
            else:
                new_data = self.data.model_dump()
                new_data.update(updates)
                self._data = self.data.__class__.model_validate(new_data)

    def get_set_file_variable(
        self, key: str, value: str | Path = None, default: str = None
    ) -> Optional[Path]:
        """Return absolute file path for a config file key.

        If *value* is provided it is saved to config and returned as an absolute
        path.  Otherwise the value already in config (or *default*) is used.
        """
        root_path = self.model.root.path.resolve()

        if isinstance(value, str):
            value = Path(value)

        if value is not None:
            full_path = (root_path / value).resolve() if not value.is_absolute() else value
            try:
                self.set(key, full_path.relative_to(root_path).as_posix())
            except ValueError:
                self.set(key, full_path.as_posix())
            return full_path

        config_value = self.get(key)
        if config_value is not None:
            value_path = Path(config_value)
        elif default is not None:
            value_path = Path(default)
            self.set(key, default)
        else:
            return None

        if not value_path.is_absolute():
            return (root_path / value_path).resolve()
        return value_path


def _convert_to_number(value):
    """Convert string to int or float where possible."""
    try:
        if isinstance(value, str):
            value = value.strip()
        num = float(value)
        return int(num) if num.is_integer() else num
    except (ValueError, TypeError):
        return value
