from matrix_zaff.levels import zaff_level, zaff_level_manifest


def test_certified_hierarchy_is_complete_and_nested() -> None:
    manifest = zaff_level_manifest()

    assert manifest["version"] == "5.0.0"
    assert [level["id"] for level in manifest["levels"]] == [
        "ZAFF0",
        "ZAFF1",
        "ZAFF2",
        "ZAFF3",
        "ZAFF4",
    ]
    assert [level["extends"] for level in manifest["levels"]] == [
        None,
        "ZAFF0",
        "ZAFF1",
        "ZAFF2",
        "ZAFF3",
    ]
    assert manifest["radial_form_contract"]["all_levels"].startswith(
        "one permanent inverse-distance stretch family"
    )


def test_zaff4_and_reactive_extension_are_unambiguous() -> None:
    manifest = zaff_level_manifest()

    assert zaff_level("Z4")["id"] == "ZAFF4"
    assert zaff_level("ZAFF-rank-complete")["extends"] == "ZAFF3"
    assert manifest["reactive_extension"]["ownership"].endswith(
        "is not a sixth ZAFF level"
    )
    assert set(manifest["downstream_consumers"]) == {
        "LINK_SENTINEL",
        "TRINITY",
        "LOCK",
        "KINETICS",
    }
