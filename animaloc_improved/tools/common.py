import json

import numpy as np

SPECIES_MAP = {
    0: "(background)",
    1: "Alcelaphinae",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}


class NumpyEncoder(json.JSONEncoder):
    """Use for value of cls argument in json.dumps(obj, cls=...)."""

    def default(self, obj: object) -> object:
        """Handle some extra classes."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)

        return super().default(obj)
