import numpy as np


class AstrobiologyEngine:
    @staticmethod
    def calculate_esi(radius: float | None, teq: float | None) -> float:
        if radius is None or teq is None or radius <= 0 or teq <= 0:
            return 0.0

        try:
            r_diff = abs((radius - 1.0) / (radius + 1.0))
            t_diff = abs((teq - 255.0) / (teq + 255.0))

            if r_diff > 1.0 or t_diff > 1.0:
                term_r = max(0.0, 1.0 - r_diff)
                term_t = max(0.0, 1.0 - t_diff)
            else:
                term_r = 1.0 - r_diff
                term_t = 1.0 - t_diff

            esi_r = term_r**0.57
            esi_t = term_t**1.07
            return float(np.sqrt(esi_r * esi_t))
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def is_habitable_zone(teq: float | None) -> bool:
        if teq is None:
            return False
        return 200.0 <= teq <= 320.0

    @staticmethod
    def is_rocky(radius: float | None) -> bool:
        if radius is None:
            return False
        return radius < 1.6

    @staticmethod
    def assign_category(esi_percent: float, is_habitable: bool, is_rocky: bool) -> str:
        if esi_percent >= 75.0 and is_habitable and is_rocky:
            return "Землеподобный супер-кандидат"
        if is_habitable:
            return "В зоне жизни (субнептун/газовый)"
        if is_rocky:
            return "Каменистая суперземля вне зоны жизни"
        return "Газовый гигант / горячий юпитер"

    @classmethod
    def compute_all_metrics(
        cls, radius: float | None, teq: float | None
    ) -> dict[str, float | bool | str]:
        raw_esi = cls.calculate_esi(radius, teq)
        esi_percent = round(raw_esi * 100.0, 2)
        habitable = cls.is_habitable_zone(teq)
        rocky = cls.is_rocky(radius)
        category = cls.assign_category(esi_percent, habitable, rocky)

        return {
            "esi": round(raw_esi, 4),
            "esi_percent": esi_percent,
            "is_habitable_zone": habitable,
            "is_rocky": rocky,
            "planet_category": category,
        }
