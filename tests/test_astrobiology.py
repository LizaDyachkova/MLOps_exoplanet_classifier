from src.astrobiology import AstrobiologyEngine


def test_calculate_esi_edge_cases():
    assert AstrobiologyEngine.calculate_esi(None, 255.0) == 0.0
    assert AstrobiologyEngine.calculate_esi(1.0, None) == 0.0
    assert AstrobiologyEngine.calculate_esi(-1.0, 255.0) == 0.0
    assert AstrobiologyEngine.calculate_esi(1.0, -10.0) == 0.0
    assert AstrobiologyEngine.calculate_esi(0.0, 0.0) == 0.0


def test_calculate_esi_earth_twin():
    esi = AstrobiologyEngine.calculate_esi(1.0, 255.0)
    assert abs(esi - 1.0) < 1e-4


def test_calculate_esi_extreme_values():
    esi = AstrobiologyEngine.calculate_esi(100.0, 5000.0)
    assert 0.0 < esi < 0.2


def test_is_habitable_zone():
    assert not AstrobiologyEngine.is_habitable_zone(None)
    assert AstrobiologyEngine.is_habitable_zone(200.0)
    assert AstrobiologyEngine.is_habitable_zone(255.0)
    assert AstrobiologyEngine.is_habitable_zone(320.0)
    assert not AstrobiologyEngine.is_habitable_zone(199.9)
    assert not AstrobiologyEngine.is_habitable_zone(320.1)


def test_is_rocky():
    assert not AstrobiologyEngine.is_rocky(None)
    assert AstrobiologyEngine.is_rocky(1.0)
    assert AstrobiologyEngine.is_rocky(1.59)
    assert not AstrobiologyEngine.is_rocky(1.6)
    assert not AstrobiologyEngine.is_rocky(2.5)


def test_assign_category():
    assert (
        AstrobiologyEngine.assign_category(80.0, is_habitable=True, is_rocky=True)
        == "Землеподобный супер-кандидат"
    )
    assert (
        AstrobiologyEngine.assign_category(60.0, is_habitable=True, is_rocky=False)
        == "В зоне жизни (субнептун/газовый)"
    )
    assert (
        AstrobiologyEngine.assign_category(50.0, is_habitable=False, is_rocky=True)
        == "Каменистая суперземля вне зоны жизни"
    )
    assert (
        AstrobiologyEngine.assign_category(20.0, is_habitable=False, is_rocky=False)
        == "Газовый гигант / горячий юпитер"
    )


def test_compute_all_metrics():
    metrics = AstrobiologyEngine.compute_all_metrics(1.0, 255.0)
    assert metrics["esi"] == 1.0
    assert metrics["esi_percent"] == 100.0
    assert metrics["is_habitable_zone"] is True
    assert metrics["is_rocky"] is True
    assert metrics["planet_category"] == "Землеподобный супер-кандидат"
