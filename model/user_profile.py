"""
User profile representation, validation and BMI.

BMI is shown for information only. It is NOT used to select exercises,
because BMI alone cannot describe fitness, body composition or health.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from model.config import (DURATIONS, FITNESS_LEVELS, GOALS, TARGET_MUSCLES,
                          USER_EQUIPMENT, USER_EXERCISE_TYPES, WORKOUT_DAYS)

AGE_RANGE = (14, 80)
HEIGHT_RANGE_CM = (120.0, 230.0)
WEIGHT_RANGE_KG = (30.0, 250.0)


@dataclass
class UserProfile:
    age: int
    height_cm: float
    weight_kg: float
    fitness_level: str
    goal: str
    target_muscle: str
    equipment: str
    duration: int
    days_per_week: int
    exercise_type: str

    # ------------------------------------------------------------------ BMI
    @property
    def bmi(self) -> float:
        """BMI = weight (kg) / height (m)^2"""
        height_m = self.height_cm / 100.0
        return round(self.weight_kg / (height_m ** 2), 1)

    @property
    def bmi_category(self) -> str:
        """WHO adult BMI categories (informational only)."""
        b = self.bmi
        if b < 18.5:
            return "Underweight"
        if b < 25.0:
            return "Normal range"
        if b < 30.0:
            return "Overweight"
        return "Obese range"

    # ----------------------------------------------------------- helpers
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: data[k] for k in fields})


def validate_profile_input(data: dict) -> list[str]:
    """
    Validate raw form input. Returns a list of human-friendly error messages
    (empty list = valid). Never raises.
    """
    errors: list[str] = []

    def _num(key: str, label: str, lo: float, hi: float, integer: bool = False):
        value = data.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            return
        if value != value:  # NaN
            errors.append(f"{label} must be a number.")
            return
        if integer and value != int(value):
            errors.append(f"{label} must be a whole number.")
        if not lo <= value <= hi:
            errors.append(f"{label} must be between {lo:g} and {hi:g}.")

    _num("age", "Age", *AGE_RANGE, integer=True)
    _num("height_cm", "Height (cm)", *HEIGHT_RANGE_CM)
    _num("weight_kg", "Weight (kg)", *WEIGHT_RANGE_KG)

    choices = {
        "fitness_level": ("Fitness level", FITNESS_LEVELS),
        "goal": ("Primary goal", GOALS),
        "target_muscle": ("Target muscle group", TARGET_MUSCLES),
        "equipment": ("Available equipment", USER_EQUIPMENT),
        "duration": ("Workout duration", DURATIONS),
        "days_per_week": ("Workout days per week", WORKOUT_DAYS),
        "exercise_type": ("Preferred exercise type", USER_EXERCISE_TYPES),
    }
    for key, (label, options) in choices.items():
        value = data.get(key)
        if value is None or value == "":
            errors.append(f"Please select your {label.lower()}.")
        elif value not in options:
            errors.append(f"'{value}' is not a valid option for {label.lower()}.")
    return errors


def build_profile(data: dict) -> tuple[UserProfile | None, list[str]]:
    """Validate and build a profile in one step."""
    errors = validate_profile_input(data)
    if errors:
        return None, errors
    profile = UserProfile(
        age=int(float(data["age"])),
        height_cm=float(data["height_cm"]),
        weight_kg=float(data["weight_kg"]),
        fitness_level=data["fitness_level"],
        goal=data["goal"],
        target_muscle=data["target_muscle"],
        equipment=data["equipment"],
        duration=int(data["duration"]),
        days_per_week=int(data["days_per_week"]),
        exercise_type=data["exercise_type"],
    )
    return profile, []
