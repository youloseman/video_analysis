---
slug: "field-cda"
title: "Field CdA: A Breeze You Can't Feel Outweighs Any Helmet"
description: "A helmet swap is worth 0.002-0.010 m² of CdA. A 1 m/s headwind you didn't notice moves one ride by 0.058. We worked out what a power-meter CdA can see, what it can't, and the protocol that makes it see more."
category: "bike-aero"
sport: "bike"
read_time: 6
published: "2026-11-27"
featured: true
kind: "research"
cover: "/media/academy/field-cda/cover.webp"
cover_alt: "A breeze you can't feel outweighs any helmet: a helmet swap is 0.002 to 0.010 m² of CdA, an unfelt 1 m/s headwind on one ride 0.058, the same headwind out and back under 0.001, bars to scale"
cover_credit: "Chart: Flapp"
cover_fit: "contain"
card: "wide"
cta: "Logged a CdA? Film the position that made it. A side-on photo from the trainer or a race is enough to start"
cta_button: "Analyze my position"
cta_href: "/app?what=bike-photo"
cta_bar: "Have a side-on bike photo? Start there"
sources:
  - author: "Chung, R."
    year: 2012
    title: "Estimating CdA with a power meter (the virtual elevation method)"
    finding: "From power, speed and air density, rebuild the road's elevation profile; the CdA and rolling resistance that make a closed loop end where it started are the rider's. Loops and out-and-backs let wind and gradient errors cancel."
  - author: "Graae, K. V. (AeroGain)"
    year: 2026
    title: "Which time trial helmet is fastest? Velodrome data from real riders (aero-gain.com, 2 November 2026)"
    finding: "Two TT helmets per rider in one session at 45 km/h, measurement noise 1.2 W. Largest median helmet gap 2.3 W (ABUS TimeShifter); a shoulder shrug moved about 5 W."
  - author: "Cyclingnews"
    year: 2025
    title: "Aero helmet wind tunnel test, Silverstone (July 2025)"
    finding: "At 40 km/h, the fastest road helmet (Oakley Velo Mach) was 8.1 W faster than a budget helmet."
  - author: "Fintelman, D. M."
    year: 2015
    title: "Influence of cycling position and crosswinds on performance and aerodynamics (PhD thesis, University of Birmingham)"
    finding: "Frontal area 0.335, 0.318, 0.302 and 0.288 m² at torso angles of 24°, 16°, 8° and 0° (n=19)."
  - author: "van Druenen, T. & Blocken, B."
    year: 2023
    title: "Aerodynamic impact of cycling postures on drafting in single paceline configurations (Computers & Fluids, 257)"
    finding: "CdA dropped-high 0.266, dropped-low 0.231, time-trial 0.213 m² at 15 m/s."
---

A good helmet swap is worth 0.002 to 0.010 m² of CdA. A 1 m/s headwind you didn't notice moves a single ride by 0.058 m², six to thirty helmets' worth. Field CdA still works. It just measures something other than what most people ask of it.

- **One ride, one way, is a guess.** A breeze you can't feel or a road that rises 0.5% moves the number by a quarter to a third.
- **A loop takes most of that away.** The same 1 m/s headwind, out and back, leaves an error under 0.001 m².
- **Trust the difference, not the number.** Two setups in one session on one loop share almost every error. Two rides a week apart share none.

![What field CdA has to see, against what can move one ride. Changes in CdA and the watts they're worth at 35 km/h](/media/academy/field-cda/scale.webp)

## Why one ride can't see a helmet

A power-meter CdA is what's left after the maths takes power, speed, rolling resistance and the slope off your ride file. Anything it doesn't know about lands in the CdA. An unfelt 1 m/s headwind makes the air hit you at 38.6 km/h instead of 35. Drag grows with the square of air speed, so the ride reads 23% more drag than you have. A 0.5% rise you took for flat takes about 40 W at 35 km/h, and the maths has nowhere to put those watts but the drag.

A TT helmet swap is under 1% of a typical CdA. To see it, run-to-run noise has to be smaller still. AeroGain's velodrome rig, out of the wind, gets to about 1 W. An open road on one pass doesn't come close.

## What cancels and what doesn't

![The same unfelt headwind, one way and out and back. Out and back stays under a TT helmet swap up to about 2 m/s](/media/academy/field-cda/loop.webp)

| What's off | One ride, one way | Same session, setup A vs B on one loop |
|---|---|---|
| A steady 1 m/s headwind | +23% | out and back: -0.2% |
| A 0.5% rise taken for flat | +35% | a closed loop: cancels |
| Speed sensor reads 1% high | -3% | scales both setups alike |
| Power meter reads 1.5% high | +2% | scales both setups alike |
| Air 5 °C warmer than assumed | -2% | same air for both |
| Rolling resistance off by 0.001 | -6% | same shift for both at the same speed |
| Gusts, wind shifting between runs | can't be undone | repeat until runs agree |
| You moving on the bike | can't be undone | repeat until runs agree |

The bottom two rows are the real limit. Everything above them is a constant, and constants drop out when you compare two setups ridden minutes apart. That's why a CdA from today next to one from last month says little: the air, the tyres, the power meter's calibration and the wind have all changed.

[[widget:cta]]

## How to ride it

![A, B, A, B on one loop: the change is seen only when the gap between A and B is bigger than the spread between the two A runs. Example numbers](/media/academy/field-cda/protocol.webp)

1. **Ride loops or out-and-backs,** never one direction. A closed loop with no stops, the shorter the better, so each lap sees the same wind.
2. **Change one thing per run** and ride A, B, A, B in one session. If the two A runs disagree by more than the A-B gap, you haven't seen the change yet.
3. **Hold a steady speed** and use the same rolling resistance for every run. Same tyres, same pressure.
4. **Pick calm days and log the temperature,** so you can tell a real change from a weather change.
5. **Know what it can see.** On a calm loop: position changes like a low back on the drops (0.035 m²) or an aero road helmet against a vented one (0.010). Helmet against helmet, or head tilt: that takes a velodrome or a sensor that measures the air itself, such as Notio or Aerosensor.

## What a photo sees that a ride can't

The gaps between a Kona age-grouper's position and a pro's are small for a power meter and plain in a photo. In Flapp's readings of Kona 2026, the age-groupers' median torso sat at 24° and the pros' at 21°: about 0.005 m², 3 W at 35 km/h. A helmet sitting +30% above the line of the back instead of the pros' +18% is another 3-4 W. One ride can't see either. A careful loop session might see both together. One side-on photo shows each of them, and which one moved.

![Position gaps Flapp reads from one photo, against a helmet swap and an unfelt headwind. Flapp's relative estimate, not a CdA measurement](/media/academy/field-cda/position.webp)

That also covers the last row of the table. Have someone film one pass of each run from the roadside, and Flapp tells you whether B was really B: whether the head went down, or you only thought it did.

## Where Flapp fits

Flapp doesn't measure CdA. It measures the position a CdA came from: torso angle, helmet height above the back, hip over the crank, from one side-on photo or video.

On paid plans you can log your measured CdA on a bike setup, with the tool, the protocol and what you changed. Flapp puts each test against your baseline in m² and watts, but only when both were measured the same way. A single ride gets a flag, not a comparison, and the log suggests what to test next.

![The CdA log on a setup in Flapp: head tucked to the back is 0.012 m² below the baseline, about 10 W at 40 km/h. The single ride is flagged and the baseline it can't be compared with says so. Example data](/media/academy/field-cda/app-cda.webp)

**Before you buy a faster helmet, ride the loop twice in the helmet you have.** If the two runs differ by more than 1%, your test can't see a helmet yet.

## How we calculated

Every error above is our own arithmetic: a rider holding constant power on a flat road at 35 km/h in still air, CdA 0.25 m², rider and bike 84 kg, rolling resistance 0.004, air 1.2 kg/m³. We then added one error at a time, solved for the new speed, and ran the standard power equation back as if the error weren't there. Out-and-back errors use equal distances each way at the same power. The helmet, shrug and position rows come from the sources below, converted to 35 km/h.
