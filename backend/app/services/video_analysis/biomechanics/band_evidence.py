"""Why each reference range is what it is, with links to the work behind it.

John Bye (coach) asked twice why a band is "optimal". The answer used to be a
raw citation string behind a "Source" link ("Skovereng 2020 PLoS ONE (pros
38...)"), and on photos nothing at all. This is the answer in words a rider
reads, with the studies one tap away.

Three things per band:
  basis -- what kind of evidence it is, said plainly, because they are not
           equal: ``measured`` (studies measured riders/runners this way),
           ``fitters`` (fitting books and fitters, no study), ``pros`` (what
           the Kona 2026 pros rode, our own field data), ``ours`` (a Flapp
           rule, e.g. a floor set by what 2D video can read).
  why   -- two or three short sentences: what the evidence says, and where
           the band departs from it and why.
  refs  -- ids into REFS; every link there was opened and checked
           (2026-11-22). A source with no public page (Retul's fit windows
           are not published) is named in ``why`` and not linked.

Every entry names the band it explains (``lo``/``hi``). The client shows an
entry only when the band on screen is that band, so a stored report graded on
an older band never gets an explanation written for a newer one. A test holds
each ``lo``/``hi`` to the live tables (cycling_positions, sport_configs,
photo_analyzer): moving a band without rewriting its "why" fails CI.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# The works themselves. ``cite`` is what the link says; ``title`` and
# ``venue`` are shown under it.
# --------------------------------------------------------------------------
REFS: dict[str, dict[str, str]] = {
    "skovereng2020": {"cite": "Skovereng et al. 2020", "venue": "PLOS ONE",
        "title": "On the effect of changing handgrip position on joint specific power and cycling kinematics in recreational and professional cyclists",
        "url": "https://doi.org/10.1371/journal.pone.0237768"},
    "jongerius2022": {"cite": "Jongerius et al. 2022", "venue": "Journal of Biomechanics",
        "title": "The biomechanics of maintaining effective force application across cycling positions",
        "url": "https://doi.org/10.1016/j.jbiomech.2022.111103"},
    "bini2014": {"cite": "Bini et al. 2014", "venue": "European Journal of Sport Science",
        "title": "Cyclists and triathletes have different body positions on the bicycle",
        "url": "https://doi.org/10.1080/17461391.2011.654269"},
    "holliday2021": {"cite": "Holliday & Swart 2021", "venue": "Sports Medicine and Health Science",
        "title": "Anthropometrics, flexibility and training history as determinants for bicycle configuration",
        "url": "https://doi.org/10.1016/j.smhs.2021.02.007"},
    "holliday2017": {"cite": "Holliday et al. 2017", "venue": "European Journal of Sport Science",
        "title": "Static versus dynamic kinematics in cyclists: a comparison of goniometer, inclinometer and 3D motion capture",
        "url": "https://doi.org/10.1080/17461391.2017.1351580"},
    "holliday2019": {"cite": "Holliday et al. 2019", "venue": "Sports Biomechanics",
        "title": "Cycling: joint kinematics and muscle activity during differing intensities",
        "url": "https://doi.org/10.1080/14763141.2019.1640279"},
    "bouillod2016": {"cite": "Bouillod et al. 2016", "venue": "icSPORTS 2016",
        "title": "Validity and reliability of the 3D Motion Analyzer in comparison with the Vicon device for biomechanical pedalling analysis",
        "url": "https://doi.org/10.5220/0006088200630066"},
    "branco2022": {"cite": "Branco et al. 2022", "venue": "Journal of Science and Cycling",
        "title": "Does the Retül system provide reliable kinematics information for cycling analysis?",
        "url": "https://doi.org/10.28985/1322.jsc.15"},
    "ferrerroca2012": {"cite": "Ferrer-Roca et al. 2012", "venue": "Journal of Strength and Conditioning Research",
        "title": "Influence of saddle height on lower limb kinematics in well-trained cyclists: static vs. dynamic evaluation in bike fitting",
        "url": "https://doi.org/10.1519/JSC.0b013e318245c09d"},
    "binihume2016": {"cite": "Bini & Hume 2016", "venue": "Human Movement",
        "title": "A comparison of static and dynamic measures of lower limb joint angles in cycling: application to bicycle fitting",
        "url": "https://doi.org/10.1515/humo-2016-0005"},
    "bini2011": {"cite": "Bini et al. 2011", "venue": "Sports Medicine",
        "title": "Effects of bicycle saddle height on knee injury risk and cycling performance",
        "url": "https://doi.org/10.2165/11588740-000000000-00000"},
    "fintelman2014": {"cite": "Fintelman et al. 2014", "venue": "Journal of Biomechanics",
        "title": "Optimal cycling time trial position models: aerodynamics versus power output and metabolic energy",
        "url": "https://doi.org/10.1016/j.jbiomech.2014.02.029"},
    "fintelman2015": {"cite": "Fintelman et al. 2015", "venue": "Journal of Sports Sciences",
        "title": "The effect of time trial cycling position on physiological and aerodynamic variables",
        "url": "https://doi.org/10.1080/02640414.2015.1009936"},
    "maddocks2025": {"cite": "Maddocks & Walker 2025", "venue": "Sports Engineering",
        "title": "Effect of arm position on the aerodynamic drag of an individual time trial position",
        "url": "https://doi.org/10.1007/s12283-025-00495-7"},
    "uci2023": {"cite": "UCI 2023", "venue": "uci.org",
        "title": "UCI Regulations for road cycling: changes for time trials",
        "url": "https://www.uci.org/pressrelease/uci-regulations-for-road-cycling-changes-for-time-trials/1QmN4gRxrsgeQtgahAAfXr"},
    "steinmetz2024": {"cite": "Steinmetz 2024", "venue": "Slowtwitch",
        "title": "Aero evolution: Kona pro bike positions",
        "url": "https://slowtwitch.com/news/aero-evolution-kona-pro-bike-positions/"},
    "burt2014": {"cite": "Burt 2014", "venue": "Bloomsbury (book)",
        "title": "Bike Fit: Optimise Your Bike Position for High Performance and Injury Avoidance",
        "url": "https://openlibrary.org/books/OL28517703M"},
    "bikefit": {"cite": "BikeFit 2019", "venue": "bikefit.com",
        "title": "Handlebar adjustments and hand position in bike fitting",
        "url": "https://bikefit.com/blogs/bikefit-blog/handlebar-adjustments-and-hand-position-in-bike-fitting"},
    "empfield2007": {"cite": "Empfield 2007", "venue": "Slowtwitch",
        "title": "Cockpit length (F.I.S.T. tri bike fit system)",
        "url": "https://www.slowtwitch.com/industry/cockpit-length/"},
    "heiderscheit2011": {"cite": "Heiderscheit et al. 2011", "venue": "Medicine & Science in Sports & Exercise",
        "title": "Effects of step rate manipulation on joint mechanics during running",
        "url": "https://doi.org/10.1249/MSS.0b013e3181ebedf4"},
    "folland2017": {"cite": "Folland et al. 2017", "venue": "Medicine & Science in Sports & Exercise",
        "title": "Running technique is an important component of running economy and performance",
        "url": "https://doi.org/10.1249/MSS.0000000000001245"},
    "weyand2000": {"cite": "Weyand et al. 2000", "venue": "Journal of Applied Physiology",
        "title": "Faster top running speeds are achieved with greater ground forces not more rapid leg movements",
        "url": "https://doi.org/10.1152/jappl.2000.89.5.1991"},
    "souza2016": {"cite": "Souza 2016", "venue": "Physical Medicine and Rehabilitation Clinics of North America",
        "title": "An evidence-based videotaped running biomechanics analysis",
        "url": "https://doi.org/10.1016/j.pmr.2015.08.006"},
}

BASIS_LABEL: dict[str, str] = {
    "measured": "Measured in studies",
    "fitters": "Bike-fit guidance",
    "pros": "What the pros ride",
    "ours": "Our rule",
}

# Academy articles that explain a band at length (slug -> link text).
ARTICLES: dict[str, str] = {
    "kona": "kona-2026-pro-bike-positions",
    "cadence": "running-cadence",
    "trunk_lean": "trunk-lean",
    "contact": "ground-contact-time",
    "vosc": "vertical-oscillation",
    "overstride": "overstriding",
}

_KNEE_BDC_WHY = (
    "Fit windows put the knee 35-45° short of straight at the bottom of the "
    "stroke. Studies that filmed riders pedalling read 141-148° on average and "
    "recommend 140-150°. We keep the fit window until we have checked our camera "
    "reading against fit markers, which read the knee a few degrees differently"
)
_KNEE_BDC_REFS = ["ferrerroca2012", "binihume2016", "bini2011"]

_KNEE_TDC_WHY = (
    "A fit-window range for the top of the stroke. Fit vendors use about "
    "67-73°; there is no study we can link that sets it"
)

_ROAD_HIP_WHY = (
    "Bike-fit guidance: Burt recommends 55-65° between torso and thigh at the "
    "top of the stroke. No study measures the hip this way, so read it as a "
    "fitter's rule. More open than the band is comfort, not a fault"
)

_AERO_KNEE_WHY = (
    "The time-trial fit window: 35-42° short of straight at the bottom. Studies "
    "of riders pedalling read 141-148° on average; aero riders sit a little "
    "more forward, so the window is slightly narrower"
)

_AERO_HIP_WHY = (
    "Bike-fit guidance for aero bars. A lower torso closes the hip, and in a "
    "study of 19 riders it cost power. Below about 45° fitters also watch for a "
    "kinked iliac artery in hard-training riders. More open than the band is "
    "comfort, not a fault"
)

_FOREARM_WHY = (
    "The UCI raised the cap on extension tilt from 15° to 30° in 2023, and the "
    "top of the band is that cap. A CFD study found hands higher with narrower "
    "elbows cut drag area by about 11%. Kona pros add 10-20° of bar tilt; the "
    "2026 field rode a median 25°"
)

_AERO_SHOULDER_WHY = (
    "Fitters aim for about 90° on aero bars, the elbow under the shoulder "
    "carrying the weight. More closed than 90° is the rider's choice and costs "
    "little; much more open can become an issue. No study sets a best shoulder "
    "angle"
)

_AERO_ELBOW_WHY = (
    "Fitters set the elbow near 90° on the pads, so the upper arm stacks under "
    "the shoulder and carries the weight. The band is wide because pad setups "
    "vary. No study sets a best elbow angle on aero bars"
)

# (sport, position, field) -> entry. ``position`` is None for running. The
# field is the name the report already uses for the reading (reference_bands'
# keys; photo rows are mapped onto them client-side).
EVIDENCE: dict[tuple[str, str | None, str], dict[str, Any]] = {
    # ---------------------------------------------------------------- road
    ("bike", "road_hoods", "trunk_angle_avg"): {
        "lo": 32, "hi": 50, "basis": "measured",
        "why": (
            "Trained riders on the hoods sit at 37-42° in three studies of about "
            "110 riders, pros a little lower than club riders. The band reaches "
            "down to 32° for racers and up to 50° so a relaxed rider is not told "
            "to get lower. Lower than the band is a racing position, not a fault"
        ),
        "refs": ["skovereng2020", "jongerius2022", "bini2014"],
    },
    ("bike", "road_hoods", "elbow_angle_avg"): {
        "lo": 145, "hi": 170, "basis": "measured",
        "why": (
            "Riders on the hoods measured 157-161° standing still on the bike and "
            "145-149° while riding, bending more as the effort rose. Most of the "
            "data comes from one lab and male riders"
        ),
        "refs": ["holliday2021", "holliday2017", "holliday2019", "bouillod2016"],
    },
    ("bike", "road_hoods", "shoulder_angle_avg"): {
        "lo": 70, "hi": 90, "basis": "measured",
        "why": (
            "Two studies measured this angle the way we do and read 76-78° on the "
            "hoods. Fitters aim for about 90° on the fit stand. Weak evidence: "
            "thirteen riders in all. More closed than the band usually means a "
            "short reach, and costs little"
        ),
        "refs": ["bouillod2016", "branco2022", "bikefit"],
    },
    ("bike", "road_hoods", "knee_at_bdc"): {
        "lo": 135, "hi": 145, "basis": "measured",
        "why": _KNEE_BDC_WHY, "refs": _KNEE_BDC_REFS,
    },
    ("bike", "road_hoods", "knee_at_tdc"): {
        "lo": 65, "hi": 75, "basis": "fitters", "why": _KNEE_TDC_WHY, "refs": [],
    },
    ("bike", "road_hoods", "hip_at_tdc"): {
        "lo": 55, "hi": 70, "basis": "fitters", "why": _ROAD_HIP_WHY, "refs": ["burt2014"],
    },
    ("bike", "road_drops", "trunk_angle_avg"): {
        "lo": 22, "hi": 38, "basis": "measured",
        "why": (
            "The one study that measured the same riders on the hoods and in the "
            "drops read the drops 8-10° lower: pros 28°, club riders 34°. The band "
            "is the hoods band moved down by that much"
        ),
        "refs": ["skovereng2020"],
    },
    ("bike", "road_drops", "knee_at_bdc"): {
        "lo": 135, "hi": 145, "basis": "measured",
        "why": _KNEE_BDC_WHY, "refs": _KNEE_BDC_REFS,
    },
    ("bike", "road_drops", "knee_at_tdc"): {
        "lo": 65, "hi": 75, "basis": "fitters", "why": _KNEE_TDC_WHY, "refs": [],
    },
    ("bike", "road_drops", "hip_at_tdc"): {
        "lo": 52, "hi": 68, "basis": "fitters",
        "why": (
            "Bike-fit guidance (Burt: 55-65° on the hoods), a few degrees more "
            "closed in the drops. No study measures the hip this way. More open "
            "than the band is comfort, not a fault"
        ),
        "refs": ["burt2014"],
    },
    # ---------------------------------------------------------------- TT
    ("bike", "tt_aero", "trunk_angle_avg"): {
        "lo": 15, "hi": 25, "basis": "pros",
        "why": (
            "The 32 pros we measured at Kona 2026 rode a median 20°, the middle "
            "half 18-22°, none above 27°. A model of 19 riders found a fully flat "
            "torso is not the fastest: power drops faster than drag. Flatter than "
            "15° costs no points, if you can hold it and still push"
        ),
        "refs": ["fintelman2014"], "article": "kona",
    },
    ("bike", "tt_aero", "elbow_angle_avg"): {
        "lo": 75, "hi": 110, "basis": "fitters", "why": _AERO_ELBOW_WHY, "refs": [],
    },
    ("bike", "tt_aero", "shoulder_angle_avg"): {
        "lo": 80, "hi": 100, "basis": "fitters", "why": _AERO_SHOULDER_WHY,
        "refs": ["empfield2007"], "article": "kona",
    },
    ("bike", "tt_aero", "knee_at_bdc"): {
        "lo": 138, "hi": 145, "basis": "measured", "why": _AERO_KNEE_WHY, "refs": _KNEE_BDC_REFS,
    },
    ("bike", "tt_aero", "knee_at_tdc"): {
        "lo": 60, "hi": 72, "basis": "fitters",
        "why": "A fit-window range for the top of the stroke, a little more closed on aero bars. There is no study we can link that sets it",
        "refs": [],
    },
    ("bike", "tt_aero", "hip_at_tdc"): {
        "lo": 42, "hi": 60, "basis": "fitters", "why": _AERO_HIP_WHY, "refs": ["burt2014", "fintelman2015"],
    },
    ("bike", "tt_aero", "forearm_tilt_avg"): {
        "lo": 5, "hi": 30, "basis": "measured", "why": _FOREARM_WHY,
        "refs": ["uci2023", "maddocks2025", "steinmetz2024"], "article": "kona",
    },
    # ---------------------------------------------------------- triathlon
    ("bike", "triathlon", "trunk_angle_avg"): {
        "lo": 20, "hi": 25, "basis": "pros",
        "why": (
            "The 32 pros we measured at Kona 2026 rode a median 20°, the middle "
            "half 18-22°, none above 27°. Triathlon guidance puts the upper end "
            "near 25° so the legs are left for the run. Flatter than 20° costs no "
            "points, if you can hold it"
        ),
        "refs": ["fintelman2014"], "article": "kona",
    },
    ("bike", "triathlon", "elbow_angle_avg"): {
        "lo": 75, "hi": 110, "basis": "fitters", "why": _AERO_ELBOW_WHY, "refs": [],
    },
    ("bike", "triathlon", "shoulder_angle_avg"): {
        "lo": 80, "hi": 105, "basis": "fitters", "why": _AERO_SHOULDER_WHY,
        "refs": ["empfield2007"], "article": "kona",
    },
    ("bike", "triathlon", "knee_at_bdc"): {
        "lo": 138, "hi": 145, "basis": "measured", "why": _AERO_KNEE_WHY, "refs": _KNEE_BDC_REFS,
    },
    ("bike", "triathlon", "knee_at_tdc"): {
        "lo": 60, "hi": 72, "basis": "fitters",
        "why": "A fit-window range for the top of the stroke, a little more closed on aero bars. There is no study we can link that sets it",
        "refs": [],
    },
    ("bike", "triathlon", "hip_at_tdc"): {
        "lo": 45, "hi": 62, "basis": "fitters",
        "why": _AERO_HIP_WHY.replace("Below about 45°", "A little more open than for TT, to save the legs for the run. Below about 45°"),
        "refs": ["burt2014", "fintelman2015"],
    },
    ("bike", "triathlon", "forearm_tilt_avg"): {
        "lo": 5, "hi": 30, "basis": "measured", "why": _FOREARM_WHY,
        "refs": ["uci2023", "maddocks2025", "steinmetz2024"], "article": "kona",
    },
    # ------------------------------------------------------------- running
    ("run", None, "cadence_spm"): {
        "lo": 170, "hi": 225, "basis": "measured",
        "why": (
            "The fault is a low cadence. Raising it 5-10% above a runner's own "
            "cuts the load on the knee and hip. There is no target above: fast "
            "runners race at 185-205, so the top is only the limit of a real reading"
        ),
        "refs": ["heiderscheit2011"], "article": "cadence",
    },
    ("run", None, "trunk_lean_avg"): {
        "lo": -2, "hi": 10, "basis": "measured",
        "why": (
            "A more upright trunk went with better economy and race times in a "
            "study of 97 runners; only a big forward lean hurt. The 2° behind "
            "vertical is room for camera tilt, not a target"
        ),
        "refs": ["folland2017"], "article": "trunk_lean",
    },
    ("run", None, "knee_max"): {
        "lo": 140, "hi": 175, "basis": "measured",
        "why": (
            "The fault is a knee nearly locked at landing, the sign of a foot "
            "reaching out ahead. Runners land with the knee near 162°. A more bent "
            "landing is not a fault: it comes with speed"
        ),
        "refs": ["heiderscheit2011", "souza2016"], "article": "overstride",
    },
    ("run", None, "knee_min"): {
        "lo": 30, "hi": 100, "basis": "ours",
        "why": (
            "The fault is a swing knee that does not fold, a long lever to swing "
            "through. Distance runners fold to about 86°, faster runners further. "
            "Above 100° costs points; the floor is how far a sprinter folds, not "
            "a target"
        ),
        "refs": ["folland2017"],
    },
    ("run", None, "ground_contact_ms"): {
        "lo": 100, "hi": 270, "basis": "measured",
        "why": (
            "Endurance runners spend 190-303 ms on the ground, 246 on average, and "
            "shorter went with faster times. It rises at an easy pace, so an easy "
            "run can read long. A short contact is not a fault"
        ),
        "refs": ["folland2017"], "article": "contact",
    },
    ("run", None, "flight_time_ms"): {
        "lo": 80, "hi": 150, "basis": "measured",
        "why": (
            "The time with both feet off the ground. Faster runners push harder "
            "into the ground rather than moving the legs faster, which shows as "
            "more time in the air. Recreational runners read about 100-140 ms"
        ),
        "refs": ["weyand2000"],
    },
    ("run", None, "vertical_oscillation_cm"): {
        "lo": 6, "hi": 10, "basis": "measured",
        "why": (
            "How far the body rises and falls each step. A 10% higher cadence "
            "brought it from about 10.7 to 8.7 cm; less bounce generally costs "
            "less energy"
        ),
        "refs": ["heiderscheit2011", "folland2017"], "article": "vosc",
    },
}


def _entry_out(entry: dict[str, Any]) -> dict[str, Any]:
    out = {
        "lo": entry["lo"], "hi": entry["hi"],
        "basis": entry["basis"], "basis_label": BASIS_LABEL[entry["basis"]],
        # A paragraph, so a full stop ends it (the data reads cleaner without).
        "why": entry["why"] if entry["why"].endswith(".") else entry["why"] + ".",
        "refs": [dict(REFS[r], id=r) for r in entry.get("refs", []) if r in REFS],
    }
    if entry.get("article"):
        out["article"] = "/academy/" + ARTICLES[entry["article"]]
    return out


def evidence_catalog() -> dict[str, Any]:
    """{sport: {position|"_": {field: entry}}} for the client."""
    out: dict[str, Any] = {}
    for (sport, position, field), entry in EVIDENCE.items():
        out.setdefault(sport, {}).setdefault(position or "_", {})[field] = _entry_out(entry)
    return out


def band_evidence(sport: str, position: str | None, field: str) -> dict[str, Any] | None:
    """The entry for one band, or None."""
    entry = EVIDENCE.get((sport, position if sport == "bike" else None, field))
    return _entry_out(entry) if entry else None
