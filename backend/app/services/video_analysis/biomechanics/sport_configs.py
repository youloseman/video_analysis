"""Reference values and configuration for sport-specific analysis.

Cycling ranges carry inline ``source`` fields / citations (Retul,
Bini, Burt, Fintelman). Running and swimming ranges are annotated
below with their primary literature; where a value reflects coaching
consensus rather than a single controlled study it is labelled as
such, rather than implying a citation that does not exist.

Running references:
  - Novacheck 1998, "The biomechanics of running" (Gait & Posture)
    -- canonical gait-phase knee-angle ranges.
  - Heiderscheit et al. 2011, "Effects of step rate manipulation on
    joint mechanics during running" (Med Sci Sports Exerc) -- ~180 spm
    target; +5-10% cadence reduces load.
  - Folland et al. 2017, "Running Technique is an Important Component
    of Running Economy and Performance" (MSSE) -- lower vertical
    oscillation and modest trunk lean track better economy.
  - Hunter et al. 2004 / Weyand et al. 2000 -- ground contact time:
    elite ~200 ms, recreational up to ~300 ms.
"""

# Running reference ranges (optimal technique). Targets are the same
# for all athlete levels (these are efficiency targets); only severity
# grading scales by level -- see running_action_plan_builder.py.
# Reference ranges reconciled against a 2024 literature review (values below are
# our INTERNAL joint angles, 180 deg = straight; papers report FLEXION from
# straight, so internal = 180 - flexion).
RUNNING_REFERENCE = {
    # deg forward lean from vertical. Folland 2017: a MORE UPRIGHT trunk
    # correlates with better performance/economy -- only extreme forward lean is
    # a clear negative -- so the band stays tolerant of upright postures. (Folland
    # measured trunk angle vs quiet standing; ours is vs vertical.)
    "trunk_lean": (2, 10),
    # steps/min. Heiderscheit 2011: preferred ~172.6; a +5-10% increase reduces
    # knee/hip load. "180" is a rule of thumb, not a strict optimum (contested).
    "cadence_spm": (170, 190),
    # internal knee angle at footstrike. Heiderscheit 2011: ~17.8 deg flexion at
    # contact = ~162 deg internal; near-locked (>~176) => overstriding.
    "knee_at_initial_contact": (160, 175),
    # internal at peak stance flexion. Heiderscheit 2011: ~46 deg flexion =
    # ~134 deg internal (more flexion at higher cadence).
    "knee_at_midstance": (130, 150),
    # internal at peak swing flexion. Folland 2017: ~94 deg flexion = ~86 deg
    # internal (shortens the swing lever).
    "knee_at_swing": (80, 100),
    # deg (~90 deg arm carry). Napier, "Science of Running": elbows ~90 deg;
    # restricting arm swing raises metabolic cost.
    "elbow_angle": (85, 100),
    # cm COM vertical excursion. Heiderscheit 2011: ~8.7 cm at higher cadence
    # (10.7 -> 8.7 across preferred-10% -> +10%); lower is generally economical.
    "vertical_oscillation_cm": (6, 10),
    # ms stance duration. Folland 2017: mean 246, range 190-303 across endurance
    # runners; SPEED-DEPENDENT (slower pace = longer contact), so easy runs read
    # high legitimately. Shorter correlates with performance; elite ~200.
    "ground_contact_ms": (180, 270),
    # ms aerial phase (Weyand 2000; recreational ~100-140).
    "flight_time_ms": (80, 150),
    # foot-ahead/leg-length at contact; <0.15 = foot near under hip
    # (Heiderscheit 2011; Souza 2016).
    "overstride_ratio": (0.0, 0.15),
}

# Cycling reference ranges (bike fit)
CYCLING_REFERENCE = {
    "knee_at_bdc": (140, 150),           # bottom dead center
    "knee_at_tdc": (65, 75),             # top dead center
    "trunk_angle": (30, 45),             # degrees (road bike)
    "elbow_angle": (90, 140),
    "hip_angle_max": (55, 65),           # avg hip angle (above = comfort/OK)
}

# Swimming reference ranges (freestyle technique).
#
# Sources: these are predominantly coaching-consensus ranges rather
# than values from controlled in-water motion-capture studies, and
# should be read that way. Primary references:
#   - Maglischo 2003, "Swimming Fastest" -- high-elbow catch (early
#     vertical forearm), streamline, body line, kick amplitude.
#   - Counsilman 1968, "The Science of Swimming" -- catch mechanics.
#   - Craig & Pendergast 1979, "Relationships of stroke rate, distance
#     per stroke, and velocity..." (MSSE) -- stroke-rate framework.
#   - Psycharakis & Sanders 2010 -- body roll in front crawl (~40-60 deg).
#
# IMPORTANT: these targets are NOT validated against MediaPipe's
# accuracy in/under water, which is the weakest link in the swim
# pipeline (BlazePose is trained on land poses). ``body_rotation`` in
# particular is retained for reference only -- the 2D lateral analyzer
# explicitly does NOT measure roll (see swimming_analyzer.py).
SWIMMING_REFERENCE = {
    "elbow_at_catch": (90, 120),         # deg, early vertical forearm (Maglischo 2003; Counsilman 1968)
    "body_rotation": (40, 60),           # deg body roll (Psycharakis & Sanders 2010) -- NOT measured in 2D
    "streamline": (0, 10),               # deg body alignment (Maglischo 2003)
    "stroke_rate_spm": (50, 65),         # strokes/min (Craig & Pendergast 1979; coaching consensus)
    "elbow_at_pull": (80, 100),          # deg (coaching consensus)
    "entry_angle": (10, 25),             # deg from vertical at hand entry (coaching consensus)
    "head_position": (0, 15),            # deg head lift from body line (Maglischo 2003)
    "kick_amplitude": (20, 40),          # deg knee-angle range during kick (coaching consensus)
}


# Pelvic stability reference ranges (rear-view cycling)
PELVIC_STABILITY_REFERENCE = {
    "pelvic_amplitude_deg": (0, 5),       # total lateral rock (excellent < 3, normal < 5)
    "asymmetry_pct": (0, 20),             # left-right symmetry
    "shoulder_amplitude_deg": (0, 4),     # upper body lateral sway
}


# Running rear-view (frontal-plane) gait analysis thresholds (multi-tier).
#
# Sources:
#   - Bramah et al. 2018, "Is There a Pathological Gait Associated With
#     Common Soft Tissue Running Injuries?" (Am J Sports Med) --
#     contralateral pelvic drop was the strongest kinematic
#     discriminator between injured and uninjured runners; injured
#     group ~+10 deg. Drives the pelvic_drop and trunk_lean tiers.
#   - Willson & Davis 2008; Powers 2010 -- dynamic knee valgus and
#     patellofemoral / lower-limb injury risk (knee_valgus tiers).
#   - Heiderscheit et al. 2011 -- cadence (shared with side-view).
RUNNING_REAR_VIEW_CONFIG = {
    "pelvic_drop": {
        "optimal": (0.0, 6.0),
        "warning": (6.0, 10.0),
        "critical": (10.0, 999.0),
    },
    "knee_valgus": {
        "optimal": (0.0, 5.0),
        "warning": (5.0, 10.0),
        "critical": (10.0, 999.0),
    },
    "trunk_lean": {
        "optimal": (0.0, 4.0),
        "warning": (4.0, 8.0),
        "critical": (8.0, 999.0),
    },
    "cadence": {
        "optimal": (170, 190),
        "warning": (160, 170),
        "critical": (0, 160),
    },
}


# ---------------------------------------------------------------------------
# Saddle diagnostics: ankle angle as diagnostic tool for saddle height
# Source: Burt 'Bike Fit' 2014; Retul guidelines; Bini et al. 2011
# ---------------------------------------------------------------------------

SADDLE_DIAGNOSTICS = {
    "saddle_too_high": {
        "signs": [
            "Knee overextension at BDC (extension >45 deg)",
            "Excessive toe-down (plantarflexion) at BDC",
            "Pelvis rocking side-to-side",
        ],
        "injuries": [
            "Hamstring strain",
            "IT-band friction (lateral knee pain)",
            "Posterior knee pain",
        ],
        "recommendation": (
            "Lower saddle by 5mm increments. "
            "Reassess knee angle at BDC after each adjustment."
        ),
        "source": "Burt 'Bike Fit' 2014; Retul guidelines",
    },
    "saddle_too_low": {
        "signs": [
            "Excessive knee flexion at BDC (extension <30 deg)",
            "Excessive heel-down (dorsiflexion) at BDC",
        ],
        "injuries": [
            "Anterior knee pain (patellofemoral)",
            "High compressive forces on kneecap",
            "Medial (inside) knee pain",
        ],
        "recommendation": (
            "Raise saddle by 5mm increments. "
            "Target knee extension 35-45 deg (dynamic) at BDC."
        ),
        "source": "Bini et al. 2011; Burt 'Bike Fit' 2014",
    },
    "saddle_too_forward": {
        "signs": [
            "Knee far forward of pedal spindle at 3 o'clock",
            "Excessive weight on hands/wrists",
        ],
        "injuries": [
            "Anterior knee pain",
            "Ulnar neuropathy (hand numbness)",
        ],
        "recommendation": (
            "Move saddle back 5mm at a time. "
            "Note: KOPS method is unreliable (Bontrager). Use dynamic measurement."
        ),
        "source": "Bontrager 'The Myth of KOPS'; Burt 'Bike Fit' 2014",
    },
    "saddle_too_back": {
        "signs": [
            "Effectively increases saddle height",
            "Rider stretched out, over-extended",
        ],
        "injuries": [
            "Hamstring strain",
            "Posterior knee pain",
            "Lower back pain from overstretched tissues",
        ],
        "recommendation": (
            "Move saddle forward 5mm. Check that reach is comfortable."
        ),
        "source": "Burt 'Bike Fit' 2014",
    },
}

# ---------------------------------------------------------------------------
# Injury risk mapping by fit issue
# ---------------------------------------------------------------------------

INJURY_RISKS = {
    "saddle_too_high": {
        "knee_sign": "Knee overreaches at BDC (extension >45 deg)",
        "ankle_sign": "Excessive toe-down (plantar flexion)",
        "injuries": ["Hamstring strain", "IT-band syndrome", "Posterior knee pain"],
        "source": "Burt 'Bike Fit' 2014; Retul guidelines",
    },
    "saddle_too_low": {
        "knee_sign": "Excessive flexion at BDC (extension <30 deg)",
        "ankle_sign": "Excessive heel-down (dorsiflexion)",
        "injuries": [
            "Patellofemoral pain (anterior knee)",
            "Increased compressive forces on kneecap",
        ],
        "source": "Bini et al. 2011; Burt 'Bike Fit' 2014",
    },
    "hip_too_closed": {
        "sign": "Hip angle <45 deg in TT position",
        "injuries": [
            "Iliac artery endofibrosis (kinking)",
            "Hip flexor tightness",
            "Lower back pain",
            "Restricted breathing",
        ],
        "medical_warning": True,
        "source": "Burt 'Bike Fit' 2014; Fintelman et al. 2015",
    },
    "excessive_reach": {
        "sign": "Elbows locked out, shoulders protracted",
        "injuries": [
            "Neck/shoulder pain",
            "Hand numbness",
            "Upper back fatigue",
        ],
        "source": "General bike fit literature",
    },
}

# ---------------------------------------------------------------------------
# Ankle diagnostics for saddle height inference
# ---------------------------------------------------------------------------

ANKLE_DIAGNOSTICS = {
    "excessive_toe_down": {
        "description": "Excessive plantar flexion at BDC",
        "probable_cause": "Saddle too HIGH",
        "mechanism": "Body compensates by pointing toes to prevent knee overreaching",
        "source": "Bike fit literature consensus",
    },
    "excessive_heel_down": {
        "description": "Excessive dorsiflexion at BDC",
        "probable_cause": "Saddle too LOW",
        "mechanism": "Body absorbs low saddle by dropping heel to preserve knee angle",
        "source": "Bike fit literature consensus",
    },
}

# ---------------------------------------------------------------------------
# Rider type adjustments for AI Coach context
# ---------------------------------------------------------------------------

RIDER_TYPE_ADJUSTMENTS = {
    "recreational": {
        "philosophy": "Comfort over aerodynamics. Endurance geometry.",
        "note": "Super-aero flat-backed position is impossible to sustain for 8-hour sportive.",
        "saddle_height": "Conservative -- mid-range of fit window, not maximum power.",
    },
    "competitive_road": {
        "philosophy": "Balance of aero and power. Gradual progression from recreational.",
    },
    "elite_tt": {
        "philosophy": "Maximum sustainable aero. Must train in position extensively.",
        "note": "Elite riders develop extreme flexibility over years. Sudden changes cause injury.",
    },
    "adaptation_rules": {
        "principle": "NEVER make sudden large changes. Incremental adjustments only.",
        "saddle_height": "5mm per adjustment, ride 30min before reassessing.",
        "handlebar_drop": "5mm at a time over weeks.",
        "warning": "Sudden 3cm handlebar drop is the leading cause of muscle spasm and injury.",
        "timeline": "Full adaptation to aggressive position takes months to years, not days.",
        "source": "Burt 'Bike Fit' 2014; track cycling longitudinal observations",
    },
}

# ---------------------------------------------------------------------------
# Measurement confidence / validation info
# ---------------------------------------------------------------------------

MEASUREMENT_CONFIDENCE = {
    "method": "2D sagittal plane video analysis (dynamic)",
    "validated_error": "<1.25 deg for joint angles (Bini & Hume 2016)",
    "standard": "Equivalent to research-grade 2D kinematic analysis",
    "limitation": "Sagittal plane only. Frontal plane (Q-angle, valgus) requires rear view.",
    "note_for_user": (
        "Measurements are taken dynamically while you pedal, which is the gold standard "
        "method used in professional bike fitting (Retul, gebioMized)."
    ),
}


def get_reference_values(sport_type: str) -> dict[str, tuple[float, float]]:
    """Get reference ranges for a sport type."""
    refs = {
        "run": RUNNING_REFERENCE,
        "bike": CYCLING_REFERENCE,
        "swim": SWIMMING_REFERENCE,
    }
    return refs.get(sport_type, RUNNING_REFERENCE)
