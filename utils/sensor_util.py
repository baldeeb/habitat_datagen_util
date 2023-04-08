import numpy as np
import magnum as mn


def get_intrinsic_from_habitat_sensor_spec(spec):
    Ho2, Wo2 = (spec.resolution[i]/2.0 for i in range(2))
    hfov = float(mn.Rad(spec.hfov))
    fy_inv = fx_inv = Wo2 / np.tan(hfov / 2.)
    return np.array([[fx_inv,    0.,        Wo2],
                    [0.,         fy_inv,    Ho2],
                    [0.,         0.,        1.0]])
