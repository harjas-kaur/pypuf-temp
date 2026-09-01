import numpy as np

from .base import LTFArray

class XORBistableRingPUF(LTFArray):
    """
    A simulation of the XOR Bistable Ring PUF [XRHB15]_.

    The simulation is based on the findings by Xu et al. [XRHB15]_ which show that Bistable Ring PUFs can be modeled
    by linear threshold functions.
    """

    def __init__(
        self,
        n: int,
        k: int,
        weights: np.ndarray,
        temperature: float = 20,
        vdd: float = 1.35,
        T_factor: bool = None,
        V_factor: bool = None,
        scale_bias: bool = False,
    ) -> None:
        """
        Initializes a XOR Bistable Ring PUF simulation.
        :param n: Number of challenge bits.
        :param k: Number of Bistable Ring PUFs
        :param weights: Array of weight values used for simulation.
        :param temperature: Environmental temperature (default 20).
        :param vdd: Supply voltage (default 1.35).
        :param T_factor: Whether to apply temperature factor (auto-detect if None).
        :param V_factor: Whether to apply voltage factor (auto-detect if None).
        """
        if weights.shape != (k, n + 1):
            raise ValueError(f"Weights must be given as an array of shape (k, n+1) = ({k}, {n+1}), but {weights.shape}"
                             f" was given.")
        # Auto-detect T_factor and V_factor if not explicitly set
        if T_factor is None:
            T_factor = (temperature != 20)  # Enable if temperature differs from nominal
        if V_factor is None:
            V_factor = (vdd != 1.35)  # Enable if vdd differs from nominal
        super().__init__(
            weight_array=weights[:, :-1],
            bias=weights[:, -1],
            transform=LTFArray.transform_id,
            combiner=self.combiner_xor,
            temperature=temperature,
            vdd=vdd,
            T_factor=T_factor,
            V_factor=V_factor,
            scale_bias=scale_bias,
        )

class BistableRingPUF(XORBistableRingPUF):
    """
    A simulation of the Bistable Ring PUF [CCLSR11]_.

    The simulation is based on the findings by Xu et al. [XRHB15]_ which show that Bistable Ring PUFs can be modeled
    by linear threshold functions.
    """

    def __init__(
        self,
        n: int,
        weights: np.ndarray,
        temperature: float = 20,
        vdd: float = 1.35,
        T_factor: bool = None,
        V_factor: bool = None,
        scale_bias: bool = False,
    ) -> None:
        if weights.shape != (n + 1,):
            raise ValueError(f"Weights must be given as an array of length n+1 = {n+1}, but an array of shape "
                             f"{weights.shape} was given.")
        super().__init__(
            n, 1, weights.reshape((1, -1)),
            temperature=temperature,
            vdd=vdd,
            T_factor=T_factor,
            V_factor=V_factor,
            scale_bias=scale_bias,
        )