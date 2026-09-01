import numpy as np


class PhysicalFactors:
    """
    Implements variations in MUX delay due to environmental factors such as
    temperature and supply voltage using an Alpha-Power Law based scaling model.
    """

    def __init__(
        self,
        temperature: float = 20,
        vdd: float = 1.35,
        m: float = 1.5,
        alpha: float = 1.2,
        Tfactor: bool = False,
        Vfactor: bool = False
    ):
        """
        Initializes the PhysicalFactors instance.

        Parameters
        ----------
        temperature : float
            Environmental temperature in °C (default 20)

        vdd : float
            Supply voltage in volts (default 1.35)

        m : float
            Temperature mobility exponent

        alpha : float
            Velocity saturation index

        Tfactor : bool
            Enable temperature scaling

        Vfactor : bool
            Enable voltage scaling
        """

        if not (0 <= temperature <= 150):
            raise ValueError("Temperature should be between 0°C and 150°C.")

        if not (0.5 <= vdd <= 5):
            raise ValueError("VDD should be between 0.5V and 5V.")

        self.temperature = temperature
        self.vdd = vdd

        # Alpha power law parameters
        self.m = m
        self.alpha = alpha

        self.Tfactor = Tfactor
        self.Vfactor = Vfactor

        # Nominal operating points
        self.T_nom_C = 20
        self.V_nom = 1.00


    def get_alpha(self, t: float, v: float) -> float:
        """
        First-order Taylor approximation of alpha(T, Vdd)
        around nominal operating point.

        alpha(T, V) ≈ alpha_ref
                    + k_T * (T - T_ref)
                    + k_V * (V - V_nom)
        """

        # Default coefficients
        alpha_ref = self.alpha
        k_T = 0.0005
        k_V = 0.01

        delta_T = t - self.T_nom_C
        delta_V = v - self.V_nom

        alpha = alpha_ref + k_T * delta_T + k_V * delta_V

        return alpha


    def temperature_dependencies(self) -> float:
        """
        Computes delay scaling due to temperature.
        """

        m = self.m
        alpha = self.alpha

        T_nom_K = self.T_nom_C + 273.15
        V_nom = self.V_nom

        current_T_K = self.temperature + 273.15

        temp_term = np.power(current_T_K / T_nom_K, m)
        volt_term = np.power(self.vdd / V_nom, alpha)

        if volt_term == 0:
            return float("inf")

        return temp_term / volt_term


    def voltage_dependencies(self) -> float:
        """
        Computes delay scaling due to supply voltage.
        """

        m = self.m
        alpha = self.alpha

        T_nom_K = self.T_nom_C + 273.15
        V_nom = self.V_nom

        current_T_K = self.T_nom_C + 273.15

        temp_term = np.power(current_T_K / T_nom_K, m)
        volt_term = np.power(self.vdd / V_nom, alpha)

        if volt_term == 0:
            return float("inf")

        return temp_term / volt_term


    def process(self, Tfactor: bool, Vfactor: bool) -> float:
        """
        Returns combined delay scaling due to temperature and voltage.
        """

        m = self.m
        alpha = self.alpha

        T_nom_K = self.T_nom_C + 273.15
        V_nom = self.V_nom

        current_T_K = self.temperature + 273.15

        temp_term = np.power(current_T_K / T_nom_K, m)
        volt_term = np.power(self.vdd / V_nom, alpha)

        if volt_term == 0:
            return float("inf")

        if Tfactor and Vfactor:
            return temp_term / volt_term

        elif Tfactor and not Vfactor:
            return self.temperature_dependencies()

        elif not Tfactor and Vfactor:
            return self.voltage_dependencies()

        else:
            return 1.0