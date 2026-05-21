# Label Mapping Presets

This project now supports multiple class-taxonomy presets so we can train in two different modes:

1. keep the original `Ecgdata` folder taxonomy
2. map the local folders into the label space used by a pretrained open-source model

## Available Presets

### `raw19_identity`

Use all `19` local folder types directly:

- `21AvB`
- `AF`
- `AtCou`
- `AtRun`
- `ComHB`
- `EctAR`
- `EctAt`
- `EctVe`
- `Idiov`
- `JuRhy`
- `Noise`
- `PaRhy`
- `Pause`
- `SArrh`
- `SRhy`
- `SVT`
- `VT`
- `VeCou`
- `Wenck`

Best use:

- full local supervision
- later custom classifier training

## `rmxjck_rhythm6_strict`

Conservative alignment to the 6-class rhythm taxonomy used by the `rmxjck` rhythm family:

- `NSR`
- `AFIB`
- `SBR`
- `AB`
- `SVTA`
- `B`

Only relatively close local folders are retained. This is the safer preset when we want cleaner semantic matching.

## `rmxjck_rhythm6_loose`

Broader alignment to the same 6-class rhythm taxonomy. This preset uses more of the local folders so the first pretrained-aligned experiment is not starved of data.

Current broad alignment:

- `AF -> AFIB`
- `AtCou, EctAR, EctAt -> AB`
- `AtRun, SVT -> SVTA`
- `SRhy, SArrh -> NSR`
- `ComHB, JuRhy, PaRhy, Wenck -> SBR`
- `VeCou, EctVe, Idiov -> B`

Important caveat:

- this is a staging map, not a clinical one-to-one equivalence map
- it is useful for backbone alignment, not for final label claims

## `rmxjck_beat3_pseudo`

Broad pseudo-alignment to the beat taxonomy used by `rmxjck/ltaf-ecg-beats-classifier-htf`:

- `N`
- `A`
- `V`

Current broad alignment:

- atrial-like folders -> `A`
- ventricular-like folders -> `V`
- sinus-like folders -> `N`

Important caveat:

- these are folder-level pseudo-labels, not beat-level truth
- this preset is meant for a later ectopy branch, not for direct rhythm claims

## Recommendation

For the immediate next step:

1. use `rmxjck_rhythm6_loose` to align the local folders to a pretrained rhythm taxonomy
2. replace the lightweight classifier with a stronger pretrained PyTorch backbone
3. keep `raw19_identity` ready for later full local fine-tuning once the backbone is stable
