from pydantic import BaseModel, Field

class BatteryTelemetryPayload(BaseModel):
    battery_id: str = Field(..., min_length=5)
    cycle_index: int = Field(..., ge=1)
    voltage_discharge_v: float = Field(..., gt=2.5, lt=4.5)
    current_discharge_a: float = Field(..., lt=0.0)
    temperature_celsius: float = Field(..., ge=-20.0, le=60.0)

class RemainingUsefulLifePrediction(BaseModel):
    battery_id: str
    predicted_rul_cycles: int
    confidence_interval: float = Field(..., ge=0.0, le=1.0)
    anomalous_thermal_state: bool
