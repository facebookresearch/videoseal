import enum


class MetricType(enum.Enum):
    NO_REFERENCE = 0
    REFERENCE = 1

class MetricObjective(enum.Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class MetricResult:

    def __init__(self, value: float, metric_objective: MetricObjective, metric_type: MetricType, metric_name: str):
        self.value = value
        self.objective = metric_objective
        self.type = metric_type
        self.name = metric_name
        self._count = 1

    def __eq__(self, other):
        assert self.name == other.name
        return self.value == other.value
    
    def __lt__(self, other):
        assert self.name == other.name
        return self.value < other.value if self.objective == MetricObjective.MAXIMIZE else self.value > other.value

    def __add__(self, other):
        assert self.name == other.name
        count = self._count + other._count
        value = (self._count * self.value + other._count * other.value) / count

        result = MetricResult(value, self.objective, self.type, self.name)
        result._count = count
        return result

    def __repr__(self):
        return str(self.value)

    def __float__(self):
        return self.value

    def __format__(self, format_spec):
        format_spec = f"{{value:{format_spec}}}"
        return format_spec.format(value=self.value)
