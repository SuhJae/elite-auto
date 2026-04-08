# Elite Dangerous Compass Alignment Notes

These notes describe a reliable ship-alignment strategy using only `pitch` and `yaw` before engaging FSD.

## Compass Meaning

- `Filled dot`: the target is in front of the ship.
- `Hollow dot`: the target is behind the ship.
- `Filled and centered`: the ship is aligned with the target.
- `Hollow and centered`: the target is directly behind the ship.

## Core Rule

- When the dot is `filled`, move it toward the center.
- When the dot is `hollow`, move it toward the rim until it flips to `filled`, then move it toward the center.

The directional sense stays the same in both cases:

- `Top`: pitch up
- `Bottom`: pitch down
- `Left`: yaw left
- `Right`: yaw right
- `Upper-left`: pitch up + yaw left
- `Upper-right`: pitch up + yaw right
- `Lower-left`: pitch down + yaw left
- `Lower-right`: pitch down + yaw right

## Axis Cases

- `Filled top`: pitch up until centered.
- `Filled bottom`: pitch down until centered.
- `Filled left`: yaw left until centered.
- `Filled right`: yaw right until centered.

- `Hollow top`: pitch up until rim -> filled -> centered.
- `Hollow bottom`: pitch down until rim -> filled -> centered.
- `Hollow left`: yaw left until rim -> filled -> centered.
- `Hollow right`: yaw right until rim -> filled -> centered.

## Quadrant Cases

- `Filled upper-left`: pitch up and yaw left.
- `Filled upper-right`: pitch up and yaw right.
- `Filled lower-left`: pitch down and yaw left.
- `Filled lower-right`: pitch down and yaw right.

- `Hollow upper-left`: pitch up and yaw left until rim -> filled; then center.
- `Hollow upper-right`: pitch up and yaw right until rim -> filled; then center.
- `Hollow lower-left`: pitch down and yaw left until rim -> filled; then center.
- `Hollow lower-right`: pitch down and yaw right until rim -> filled; then center.

## Center Cases

- `Filled and centered`: aligned, ready to FSD.
- `Hollow and centered`: target is directly behind. Use a fixed escape policy for reliability:
  - pitch up with small pulses until the dot leaves center
  - continue following the normal hollow-dot rule

## Reliable Bot Policy

1. If the dot is `hollow and centered`, use small `pitch up` pulses until it is no longer centered.
2. If the dot is `hollow`, keep applying same-side correction to drive it outward toward the rim.
3. When the dot flips to `filled`, switch objective to centering.
4. In diagonal cases, correct the larger error first.
5. Near center, reduce to tiny pulses and prefer one axis at a time.
6. Only engage FSD when the dot is `filled` and centered.

## Practical Summary

- `Hollow`: same-side input, aim for rim.
- `Filled`: same-side input, aim for center.
- `Filled + centered`: engage FSD.

## Sources

- Frontier Quick Starter Manual:
  - https://hosting.zaonce.net/elite/website/assets/ELITE-DANGEROUS-MANUAL.pdf
- Elite Dangerous Wiki HUD/Center:
  - https://elite-dangerous.fandom.com/wiki/HUD/Center
- Frontier forum explanation of hollow vs solid compass behavior:
  - https://forums.frontier.co.uk/threads/best-way-to-align-with-marked-jump-target-how-to-find-the-target.370857/
